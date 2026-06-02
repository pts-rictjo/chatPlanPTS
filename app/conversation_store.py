import uuid

class ConversationStore:
    def __init__(self):
        self.store = {}

    def create(self) -> str:
        cid = str(uuid.uuid4())
        self.store[cid] = []
        return cid

    def append(self, cid: str, role: str, content: str):
        self.store.setdefault(cid, []).append({"role": role, "content": content})

    def get(self, cid: str) -> list:
        return self.store.get(cid, [])
