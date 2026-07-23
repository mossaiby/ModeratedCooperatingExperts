"""Moderated Cooperating Experts (MoCE): a moderator LLM decomposes a user request
into typed content blocks with a dependency DAG, and specialist "expert" LLMs
(one per block type) fill each block under strict output constraints.
"""

__version__ = "0.1.0"
