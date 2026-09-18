#!/usr/bin/env python3
"""
A final, targeted script to test the 'view: "postVotes"' hypothesis
for fetching votes for a specific post.
"""
import requests
import json
import argparse

API_URLS = {
    "EA": "https://forum.effectivealtruism.org/graphql",
    "LW": "https://www.lesswrong.com/graphql",
    "AF": "https://www.alignmentforum.org/graphql",
}

# This is our highest-confidence query based on all previous tests and research.
# It uses the 'view' parameter, which is likely required for security/indexing.
VOTES_QUERY = """
query GetPostVotes($postId: String!) {
  votes(input: {
    terms: {
      view: "postVotes",
      documentId: $postId,
      limit: 10
    }
  }) {
    results {
      _id
      documentId
      voteType
      power
      votedAt
      createdAt
    }
  }
}
"""

def main():
    """Parses arguments and runs the definitive votes query."""
    parser = argparse.ArgumentParser(
        description="Test the 'postVotes' view for a GraphQL API."
    )
    parser.add_argument(
        "-u", "--url",
        required=True,
        choices=API_URLS.keys(),
        help="The short name of the API to query."
    )
    parser.add_argument(
        "-i", "--id",
        required=True,
        help="The _id of the post to inspect (e.g., '6hy7tsB2pkpRHqazG')."
    )
    args = parser.parse_args()

    api_url = API_URLS[args.url]
    post_id = args.id
    variables = {"postId": post_id}
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AI-Safety-Feed-Schema-Explorer/4.0"
    }

    print(f"--- Running Final Test for Post ID: {post_id} on {args.url} ---")
    print("--- Using view: 'postVotes' ---")

    try:
        response = requests.post(
            api_url,
            json={"query": VOTES_QUERY, "variables": variables},
            headers=headers,
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

        print("\n--- Query Result ---")
        print(json.dumps(data, indent=2))

        if data.get("errors"):
            print("\n--- [RESULT] FAILED (GraphQL Error) ---")
            print("The query syntax is incorrect.")
        elif not data.get("data", {}).get("votes", {}).get("results"):
            print("\n--- [RESULT] SUCCESS (but no votes found) ---")
            print("The query ran, but returned no data. The view name or filter key may still be wrong.")
        else:
            print("\n--- [RESULT] SUCCESS! VOTES FOUND! ---")
            print("This is the data structure we need.")

    except requests.exceptions.RequestException as e:
        print(f"\n--- [RESULT] FAILED (HTTP Error) ---")
        print(f"Error: {e}")
        if e.response:
            print(f"Response Body: {e.response.text}")

if __name__ == "__main__":
    main()