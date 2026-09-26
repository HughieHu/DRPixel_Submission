"""The 7 experiment configs: name -> {tool_name: paradigm}. Each runs on its own GPU.
- text_only      : 1 tool (text)
- txtfig_{q8,q2,js,jm} : 2 tools (text + figure), 4 figure retrievers
- page_{q8,q2}   : 1 tool (page), 2 page retrievers (Qwen-VL 8B/2B; jina pages not ready)
"""
CONFIGS = {
    "text_only": {"search_text": "text"},
    "txtfig_q8": {"search_text": "text", "search_figure": "fig_q8"},
    "txtfig_q8_full": {"search_text": "text", "search_figure": "fig_q8_full"},   # 99k figure pool
    "txtfig_q2": {"search_text": "text", "search_figure": "fig_q2"},
    "txtfig_js": {"search_text": "text", "search_figure": "fig_js"},
    "txtfig_jm": {"search_text": "text", "search_figure": "fig_jm"},
    "page_q8":   {"search_page": "pg_q8"},
    "page_q2":   {"search_page": "pg_q2"},
    "page_js":   {"search_page": "pg_js"},
    "page_jm":   {"search_page": "pg_jm"},
    # text at other chunk lengths (vs default 256-word text_only)
    "text_c1024": {"search_text": "text_c1024"},
    "text_c4096": {"search_text": "text_c4096"},
    # page at other Qwen3-VL resolutions (vs default 602112 page_q8)
    "page_r512":  {"search_page": "pg_r512"},
    "page_r1280": {"search_page": "pg_r1280"},
    # + visit tool: search to find the paper, then visit(doc_id) to read its FULL markdown
    "text_only_with_visit": {"search_text": "text", "visit": "visit"},
    "txtfig_q8_with_visit": {"search_text": "text", "search_figure": "fig_q8", "visit": "visit"},
    # page + visit: visit feeds ALL page images of the found paper (visit_pages paradigm)
    "page_q8_with_visit": {"search_page": "pg_q8", "visit": "visit_pages"},
    # prompt-variant configs: SAME tool set as their base, only the INSTR augmentation differs
    # (see react.PROMPT_EXTRA): one-shot worked figure example / good-query guidance per tool.
    "textfig_q8_one_shot":  {"search_text": "text", "search_figure": "fig_q8"},
    "textfig_q8_opt_query": {"search_text": "text", "search_figure": "fig_q8"},
    "page_q8_opt_query":    {"search_page": "pg_q8"},
    "page_q8_nlquery":      {"search_page": "pg_q8"},
    # query-rewrite experiment: SAME retriever-only tool set as page_q8 (Qwen3-VL-8B page retriever),
    # but before EACH retrieve a separate gpt-5.4 rewrites (thought, draft query) -> ~20-word NL query.
    "page_query_rewrite_10w": {"search_page": "pg_q8"},   # launch with REWRITE_WORDS=10
    "page_query_rewrite_20w": {"search_page": "pg_q8"},
    "page_query_rewrite_30w": {"search_page": "pg_q8"},   # same rewriter, launch with REWRITE_WORDS=30
    "page_query_rewrite_40w": {"search_page": "pg_q8"},   # launch with REWRITE_WORDS=40
    # KEYWORD-style rewrite (same lengths): launch with REWRITE_STYLE=keyword + REWRITE_WORDS=N
    "page_query_rewrite_kw_10w": {"search_page": "pg_q8"},
    "page_query_rewrite_kw_20w": {"search_page": "pg_q8"},
    "page_query_rewrite_kw_30w": {"search_page": "pg_q8"},
    "page_query_rewrite_kw_40w": {"search_page": "pg_q8"},
}

# configs whose runs insert the separate-model query rewriter (run.py -> react.run_react(rewrite=...)).
REWRITE_CONFIGS = {"page_query_rewrite_10w", "page_query_rewrite_20w",
                   "page_query_rewrite_30w", "page_query_rewrite_40w",
                   "page_query_rewrite_kw_10w", "page_query_rewrite_kw_20w",
                   "page_query_rewrite_kw_30w", "page_query_rewrite_kw_40w"}
