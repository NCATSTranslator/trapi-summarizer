#!/usr/bin/env python
"""FastAPI server mirroring ui_server.py but backed by Vertex-hosted models.

Everything that varies — provider, model, generation parameters, templates, and
the listen address — comes from a configuration file, one per environment. The
endpoints, payloads, and SSE event contract are identical to ui_server.py, so
this remains a drop-in swap.

    python vertex_ui_server.py --config configurations/dev.json [configurations/local-overrides.json]

With no --config, the file is derived from $APP_ENVIRONMENT (see config.py).
The service-account private key is injected through the secrets configuration;
see vertex_creds.py.
"""
import argparse
import json
import sys

from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse

from summarizers import ui_summarizer
from summarizers import ui_tools
from summarizers import gene_nmf_utils
import config
import llm_utils
import vertex_llm


def create_app(cfg) -> FastAPI:
    client = vertex_llm.client_from_config(cfg)
    template, nmf_template = vertex_llm.templates_from_config(cfg)
    max_turns = cfg["llm"]["max_turns"]
    chunk = cfg["llm"]["stream_chunk_tokens"]

    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "'Sup", "provider": cfg["llm"]["provider"], "model": client.model}

    @app.post("/summary")
    async def create_summary(payload: dict):
        summary = ui_summarizer.create_ui_summary(payload, 0)
        print(summary)
        result = await client.run_as_loop(summary, template, llm_utils.handle_fun_call,
                                          max_turns)
        return {"response_text": result.output_text}

    @app.post("/summary-streaming")
    async def create_summary_streaming(payload: dict, request: Request):
        async def event_generator():
            summary = ui_summarizer.create_ui_summary(payload, 0)
            print(summary)
            try:
                async for event in client.run_as_loop_streaming(
                        summary, template, llm_utils.handle_fun_call,
                        1, None, max_turns, chunk):
                    yield {"event": "data", "data": json.dumps({"event": event})}
                    if await request.is_disconnected():
                        break
            except Exception as e:
                yield {"event": "error", "data": json.dumps({"error": str(e)})}
            yield {"event": "complete", "data": json.dumps({"complete": True})}

        return EventSourceResponse(event_generator())

    @app.post("/gene-summary-streaming")
    async def create_gene_summary_streaming(payload: dict, request: Request):
        async def event_generator():
            # Shrink + presummary
            if 'data' in payload and 'disease' not in payload.get('data', {}):
                shrunk = ui_tools.shrink_payload(payload, 0)
                presummary = ui_tools.create_ui_presummary(shrunk, 0)
            else:
                presummary = ui_tools.create_ui_presummary(payload, 0)

            gene_dict = gene_nmf_utils.extract_genes_from_ui_nodes(presummary['nodes'])
            nmf_result = await gene_nmf_utils.generate_nmf_presummary(
                gene_dict, presummary['disease_name'])

            if nmf_result is None:
                yield {"event": "error", "data": json.dumps({"error": "No genes found in result"})}
                yield {"event": "complete", "data": json.dumps({"complete": True})}
                return

            try:
                final_event = None
                async for event in client.run_as_loop_streaming(
                        nmf_result.presummary, nmf_template, llm_utils.handle_fun_call,
                        1, None, max_turns, chunk):
                    final_event = event
                    yield {"event": "data", "data": json.dumps({"event": event})}
                    if await request.is_disconnected():
                        break

                # Post-process: wrap with warning banner + factor listing
                if final_event:
                    wrapped = gene_nmf_utils.wrap_nmf_response(
                        final_event.get('output_text', ''), nmf_result,
                        gene_nmf_utils.DEFAULT_MIN_GENES)
                    yield {"event": "wrapped", "data": json.dumps({"response_text": wrapped})}
            except Exception as e:
                yield {"event": "error", "data": json.dumps({"error": str(e)})}
            yield {"event": "complete", "data": json.dumps({"complete": True})}

        return EventSourceResponse(event_generator())

    return app


def main():
    p = argparse.ArgumentParser(description="Vertex-backed summary server (mirror of ui_server.py)")
    p.add_argument('-c', '--config', nargs='+', metavar='FILE',
                   help='Base config file, optionally followed by an override file. '
                        f'Defaults to configurations/<${config.APP_ENV_VAR}>.json')
    args = p.parse_args()

    try:
        base, override = config.resolve_config_paths(args.config)
        cfg = config.bootstrap(base, override)
    except config.ConfigError as e:
        sys.exit(f"Configuration error: {e}")

    print(f"Configuration ({base}"
          f"{' + ' + override if override else ''}):")
    print(json.dumps(config.redacted(cfg), indent=2))

    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg["server"]["host"], port=cfg["server"]["port"])


if __name__ == '__main__':
    main()
