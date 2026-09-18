================================================================
Starting AI Safety Feed Ingestion Script at 2025-11-03 19:57:39.741027+00:00
================================================================

--- Fetching Raw Posts ---
Executing GraphQL query for 3000 posts from https://forum.effectivealtruism.org/graphql (Tag ID: oNiQsBHA3i837sySD)...
2025-11-03 14:57:39,742 - INFO - Executing GraphQL query for 3000 posts from https://forum.effectivealtruism.org/graphql (Tag ID: oNiQsBHA3i837sySD)
Successfully fetched 3000 posts from https://forum.effectivealtruism.org/graphql (Tag ID: oNiQsBHA3i837sySD).
2025-11-03 14:57:46,013 - INFO - Successfully fetched 3000 posts from https://forum.effectivealtruism.org/graphql (Tag ID: oNiQsBHA3i837sySD).
Executing GraphQL query for 3000 posts from https://www.lesswrong.com/graphql (Tag ID: yBXKqk8wEg6eM8w5y)...
2025-11-03 14:57:46,015 - INFO - Executing GraphQL query for 3000 posts from https://www.lesswrong.com/graphql (Tag ID: yBXKqk8wEg6eM8w5y)
Successfully fetched 2999 posts from https://www.lesswrong.com/graphql (Tag ID: yBXKqk8wEg6eM8w5y).
2025-11-03 14:57:56,831 - INFO - Successfully fetched 2999 posts from https://www.lesswrong.com/graphql (Tag ID: yBXKqk8wEg6eM8w5y).
Executing GraphQL query for 3000 posts from https://www.alignmentforum.org/graphql (View: top)...
2025-11-03 14:57:56,839 - INFO - Executing GraphQL query for 3000 posts from https://www.alignmentforum.org/graphql (View: top)
2025-11-03 14:57:56,986 - ERROR - Query failed for https://www.alignmentforum.org/graphql (View: top): 429 Client Error: Too Many Requests for url: https://www.alignmentforum.org/graphql
Traceback (most recent call last):
File "c:\Users\matth\projects\ai-safety-feed\scraping\ea_lw_query.py", line 769, in get_forum_posts
response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\requests\models.py", line 1026, in raise_for_status
raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 429 Client Error: Too Many Requests for url: https://www.alignmentforum.org/graphql
ERROR: Query failed for https://www.alignmentforum.org/graphql (View: top): 429 Client Error: Too Many Requests for url: https://www.alignmentforum.org/graphql
2025-11-03 14:57:56,988 - ERROR - Response status code: 429

--- Filtering Posts ---

--- Filtering 3000 EA Forum posts ---
--- Found 61 EA Forum posts meeting criteria ---
2025-11-03 14:57:56,994 - INFO - Filtered EA Forum posts: 3000 -> 61

--- Filtering 2999 LessWrong posts ---
--- Found 187 LessWrong posts meeting criteria ---
2025-11-03 14:57:56,997 - INFO - Filtered LessWrong posts: 2999 -> 187

--- Filtering 0 Alignment Forum posts ---

--- Total posts from all sources after initial filtering: 248 ---

--- Deduplicating 248 posts in memory by normalized title (keeping highest score) ---
--- Kept 243 unique posts (removed 5 lower-scoring duplicates) ---
2025-11-03 14:57:57,000 - INFO - Deduplication: Input 248, Valid w/ Title 248, Unique Output 243

--- Connecting to Database ---
2025-11-03 14:57:57,000 - INFO - Connecting to database using URL: postgresql://doadmin...
Database connection successful.
2025-11-03 14:57:57,438 - INFO - Database connection successful.
Initializing OpenAI client...
2025-11-03 14:57:57,722 - INFO - OpenAI client initialized.
Fetching existing normalized titles from database...
--> Found 553 existing titles in the database.
2025-11-03 14:57:57,817 - INFO - Fetched 553 existing titles.
Fetching already skipped normalized titles from database...
--> Found 1,428 already skipped titles in the database.
2025-11-03 14:57:57,914 - INFO - Fetched 1428 already skipped titles.

--- Starting Processing and Analysis for 243 Unique Posts ---

[1/243] Processing Post: 'How Well Does RL Scale?...' (ID: xpj6KhDM9bJybdnEe)
2025-11-03 14:57:57,915 - INFO - Processing post 1/243: ID xpj6KhDM9bJybdnEe, Title: How Well Does RL Scale?...
-> Cleaning HTML and converting to Markdown...
-> HTML cleaning successful.
-> Markdown conversion successful.
-> Performing Gemini analyses... - Generating sentence summary...
Gemini: starting call (model=gemini-2.5-flash, prompt_len=15082 chars)...
2025-11-03 14:57:58,505 - INFO - AFC is enabled with max remote calls: 10.
2025-11-03 14:57:58,666 - ERROR - Unexpected error during Gemini API call: The read operation timed out
Traceback (most recent call last):
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_exceptions.py", line 10, in map_exceptions
yield
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_backends\sync.py", line 126, in read
return self.\_sock.recv(max_bytes)
^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\ssl.py", line 1296, in recv
return self.read(buflen)
^^^^^^^^^^^^^^^^^
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\ssl.py", line 1169, in read
return self.\_sslobj.read(len)
^^^^^^^^^^^^^^^^^^^^^^
TimeoutError: The read operation timed out

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpx_transports\default.py", line 101, in map_httpcore_exceptions
yield
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpx_transports\default.py", line 250, in handle_request
resp = self.\_pool.handle_request(req)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_sync\connection_pool.py", line 268, in handle_request
raise exc
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_sync\connection_pool.py", line 251, in handle_request
response = connection.handle_request(request)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_sync\connection.py", line 103, in handle_request
return self.\_connection.handle_request(request)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_sync\http11.py", line 133, in handle_request
raise exc
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_sync\http11.py", line 111, in handle_request
) = self.\_receive_response_headers(\*\*kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_sync\http11.py", line 176, in \_receive_response_headers
event = self.\_receive_event(timeout=timeout)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_sync\http11.py", line 212, in \_receive_event
data = self.\_network_stream.read(
^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_backends\sync.py", line 124, in read
with map_exceptions(exc_map):
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\contextlib.py", line 155, in **exit**
self.gen.throw(typ, value, traceback)
File "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\Lib\site-packages\httpcore_exceptions.py", line 14, in map_exceptions
raise to_exc(exc) from exc
httpcore.ReadTimeout: The read operation timed out

The above exception was the direct cause of the following exception:
