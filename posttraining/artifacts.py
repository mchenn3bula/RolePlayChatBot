"""Save replies for the user without printing or scoring their content."""

import html
import json
from pathlib import Path


class ReplyWriter:
    def __init__(self, directory):
        directory = Path(directory)
        self.raw = (directory / "generations.jsonl").open("x", encoding="utf-8")
        try:
            self.gallery = (directory / "replies.html").open("x", encoding="utf-8")
        except Exception:
            self.raw.close()
            raise
        self.gallery.write(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
            '<title>Local replies — user review</title>'
            '<style>body{max-width:900px;margin:40px auto;padding:0 20px;font:17px/1.6 system-ui}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-top:1px solid #888;padding:20px 0}'
            'summary{cursor:pointer}</style><h1>Generated replies</h1>'
            '<p>Saved for your review. No automated content scoring or assistant review.</p>\n'
        )
        self.gallery.flush()

    def append(self, row):
        self.raw.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.raw.flush()
        prompt = row["messages"][-1]["content"]
        self.gallery.write(
            "<article><h2>" + html.escape(row["id"]) + "</h2><h3>Your input</h3><pre>"
            + html.escape(prompt) + "</pre><h3>Reply</h3><pre>" + html.escape(row["text"])
            + "</pre><details><summary>Input context and authored state</summary><pre>"
            + html.escape(json.dumps({"messages": row["messages"], "context": row.get("context")},
                                     ensure_ascii=False, indent=2)) + "</pre></details></article>\n"
        )
        self.gallery.flush()

    def close(self):
        self.gallery.write("</html>\n")
        self.gallery.close()
        self.raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
