from typing import Optional

from hw2_solution import DEFAULT_MCP_URL, score_cvs


def get_scores(mcp_url: Optional[str] = DEFAULT_MCP_URL) -> list[float]:
    return score_cvs(mcp_url=mcp_url)


if __name__ == "__main__":
    print(get_scores())
