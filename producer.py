import asyncio
import json
import websockets
from kafka import KafkaProducer
from datetime import datetime, timezone

#im goign to be using bluesky

url = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"

producer = KafkaProducer(
    bootstrap_servers="kafka:29092",
    value_serializer=lambda message: json.dumps(message).encode("utf-8")
)

async def listen():
    #open connection
    seen_ids = set()
    async with websockets.connect(url) as websocket:
        print("Connection works")
        true_flag = True

        while (true_flag):
            message = await websocket.recv()
            data = json.loads(message)
           
            commit = data.get("commit", {})
            record = commit.get("record", {})
            text = record.get("text", "")
            if commit.get("operation") != "create" or "en" not in record.get("langs", []):
                continue
            post_id = f"{data.get('did')}:{commit.get('rkey')}"
            if not text or post_id in seen_ids:
                continue
            seen_ids.add(post_id)

            
            print(f"Got post: {text[:60]}...")
              
            payload = {
                    "id": post_id,
                "text": text,
                "source": "bluesky",
                "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                }

            producer.send("topic1", payload).get(timeout=10)

            with open("data/streamed_messages.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            print("Sent to Kafka and saved!")
           

# Run the async loop
if __name__ == "__main__":
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        print("\nStopped cleanly.")

            







