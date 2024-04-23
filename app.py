import gradio as gr
from predict import generate_caption

demo = gr.Interface(fn=generate_caption,
             inputs=gr.components.Image(),
             outputs=[gr.components.Textbox(label="Generated Caption", lines=3)],
             )
demo.launch(share = True, debug = True)

