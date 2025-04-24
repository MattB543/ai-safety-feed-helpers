# backfill_embeddings.py
from openai import OpenAI
import psycopg2, os
from psycopg2 import extras  # Import the extras submodule
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
conn   = psycopg2.connect(os.environ["AI_SAFETY_FEED_DB_URL"])
cur    = conn.cursor(name="cur")      # server-side cursor

cur.execute("""
  SELECT id, title, authors, sentence_summary,
         paragraph_summary, key_implication
  FROM   content
  WHERE  embedding_short IS NULL AND embedding_full IS NULL
""")

while True:
    rows = cur.fetchmany(100)         # 100-vector batch keeps latency low
    if not rows: break
    short_texts, full_texts, ids = [], [], []
    for r in rows:
        ids.append(r[0])
        short_texts.append(r[1] or "")
        full_texts.append(
            f"{r[1]}\nAuthors: {', '.join(r[2] or [])}\n"
            f"{r[3]}\n{r[4]}\n{r[5]}"
        )
    embeds = client.embeddings.create(
        model="text-embedding-3-small",
        input=short_texts + full_texts
    ).data                              # length = 2 × batch
    shorts = embeds[:len(ids)]
    fulls  = embeds[len(ids):]
    extras.execute_batch(  # Use extras.execute_batch
        conn.cursor(),
        "UPDATE content SET embedding_short=%s, embedding_full=%s WHERE id=%s",
        [(s.embedding, f.embedding, i) for s,f,i in zip(shorts, fulls, ids)]
    )
conn.commit()
