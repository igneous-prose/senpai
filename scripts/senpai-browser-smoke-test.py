#!/usr/bin/env python3

"""Launch the OpenHands browser and read a deterministic local page."""

# OpenHands imports intentionally follow banner suppression below.

import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
os.environ.setdefault("BROWSER_USE_DISABLE_EXTENSIONS", "1")

from openhands.sdk.agent import Agent
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.llm import LLM
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.browser_use import BrowserToolSet
from pydantic import SecretStr


class SmokePage(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"<html><body>senpai-browser-ok</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), SmokePage)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as workspace:
            llm = LLM(
                model="gpt-4o-mini",
                api_key=SecretStr("not-used"),
                usage_id="browser-smoke-test",
            )
            state = ConversationState.create(
                id=uuid4(),
                agent=Agent(llm=llm, tools=[]),
                workspace=LocalWorkspace(working_dir=workspace),
            )
            tools = {tool.name: tool for tool in BrowserToolSet.create(state)}
            navigate = tools["browser_navigate"]
            content = tools["browser_get_content"]
            url = f"http://127.0.0.1:{server.server_port}"

            result = navigate.executor(navigate.action_type(url=url))
            assert not result.is_error, result.text
            result = content.executor(content.action_type())
            assert not result.is_error, result.text
            assert "senpai-browser-ok" in result.text
    finally:
        if BrowserToolSet._shared_executor is not None:
            BrowserToolSet._shared_executor.close()
        server.shutdown()
        server.server_close()

    print("senpai-browser-ok")


if __name__ == "__main__":
    main()
