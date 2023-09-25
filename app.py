import gradio as gr
from transformers import pipeline

# Create the fill-mask pipeline
fill_mask_pipe = pipeline("fill-mask", model="ayoubkirouane/FILL-MAsk-RoBERTa-base")

def fill_mask(text):
    # Make a prediction using the fill-mask pipeline
    results = fill_mask_pipe(text)
    return results[0]["sequence"]

example = ["The capital of Algeria is <mask>."]
# Create a Gradio interface
iface = gr.Interface(
    fn=fill_mask,
    inputs="text",
    outputs="text",
    examples=example,
    title="FILL-MAsk-RoBERTa-base Demo",
    description="Enter a sentence with a masked token, and the model will predict the missing word.",
)

# Launch the Gradio app
iface.launch()
