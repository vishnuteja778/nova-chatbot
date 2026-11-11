"""Step 0: check that your Groq API key works and see which models it can use."""
from dotenv import load_dotenv
from groq import Groq

load_dotenv()      # copies the values in .env into environment variables
client = Groq()    # finds GROQ_API_KEY in the environment by itself

models = client.models.list().data
print("Models your key can use:\n")
for model in sorted(models, key=lambda m: m.id):
    # context_window = the most tokens the model can see at once.
    # getattr(..., None) avoids a crash if the API ever leaves this field out.
    window = getattr(model, "context_window", None)
    size = f"{window:,} tokens" if window else "unknown"
    print(f"  {model.id:45} {model.owned_by:15} context: {size}")
