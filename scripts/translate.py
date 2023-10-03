import os
from pprint import pprint

import frontmatter
import mistune
from googletrans import Translator
from markdown_it import MarkdownIt

trans = Translator()


def find_index_md_files(parent_folder):
    index_md_files = []

    # Use a recursive search to find all "index.md" files
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file == "index.md":
                index_md_files.append(os.path.join(root, file))

    return index_md_files


def translate_to_french(path: str) -> str:
    with open(path) as f:
        return trans.translate(f.read(), src="en", dest="fr").text


def extract_plain_text_from_markdown(markdown_file_path):
    post = frontmatter.load(markdown_file_path)
    md = MarkdownIt("gfm-like")
    tokens = md.parse(post.content)

    for token in tokens:
        if token.type == "inline":
            token.content = trans.translate(token.content, src="en", dest="fr").text

    return md.renderer.render(tokens)


def mistune_impl(markdown_file_path):
    md = mistune.create_markdown(
        escape=False,
        plugins=["strikethrough", "footnotes", "table", "speedup"],
        renderer=None,
    )

    with open(markdown_file_path) as f:
        ast = md.parse(f.read())
    print(ast[1])


content = find_index_md_files("../content")
# print(extract_plain_text_from_markdown(content[0]))
print(mistune_impl(content[0]))

# res = trans.translate('Good evening sir. How may I help you?', dest='fr')
# back = trans.translate(res.text, dest='en')
# print(res.text, back.text)
