import subprocess
import struct
import pathlib
import gradio as gr
import numpy as np

this_file_path = pathlib.Path(__file__).parent
# process link to the C++ application
# add the argument to the model file to be loaded by the C++ application
process = subprocess.Popen(
    ['./bin/live_demo', 'mnist_fc128_relu_fc10_sigmoid'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=this_file_path
)


def run_inference(sketchpad_input):
    
    pil_image = sketchpad_input["composite"]
    
    input_vector = np.array(pil_image.resize((28, 28)), dtype=np.float64)
    # Check if is all zeros
    if np.all(pil_image == 0):
        return None

    input_vector = input_vector.astype(np.float64) 
    input_vector = input_vector / 255.0  # Normalize pixel values to 0-1
    input_vector = input_vector.flatten(order='C')  # Flatten the image (column-major order)

    # Send the input vector to the C++ program, as <num_items>, <item1>, <item2>, ...
    packed_size = struct.pack('Q', len(input_vector))
    packed_vector = struct.pack(f"{len(input_vector)}d", *input_vector)
    process.stdin.write(packed_size)
    process.stdin.write(packed_vector)
    process.stdin.flush()
    
    # Read the size and the result vector (array of doubles)
    packed_size = process.stdout.read(struct.calcsize('Q'))
    result_size = struct.unpack('Q', packed_size)[0]
    packed_result = process.stdout.read(result_size * struct.calcsize('d'))
    result_vector = struct.unpack(f'{result_size}d', packed_result)

    return {str(label): conf for label, conf in enumerate(list(result_vector))}


with gr.Blocks() as interface:
    gr.Markdown("# MNIST Digit Recognition")
    # Create the components and set render to false
    # so that items can be rendered in a custom layout
    result_output = gr.Label(label="Classification Result", show_label=True, num_top_classes=10, render=False)
    brush = gr.Brush(default_size=20, colors=["white"], color_mode="fixed")
    sketchpad = gr.Sketchpad(height=500, canvas_size=(400, 400), type='pil', image_mode='L', brush=brush, layers=False, transforms=[], render=False)
    with gr.Row():
        with gr.Column(min_width=500, scale=0):
            sketchpad.render()    
            run_btn = gr.Button(value="Classify Drawing")

        with gr.Column():
            result_output.render()
    
    with gr.Row():
        gr.Examples(
            examples=[
                [f"{this_file_path}/example_images/0.png"],
                [f"{this_file_path}/example_images/1.png"],
                [f"{this_file_path}/example_images/2.png"],
                [f"{this_file_path}/example_images/3.png"],
                [f"{this_file_path}/example_images/4.png"],
                [f"{this_file_path}/example_images/5.png"],
                [f"{this_file_path}/example_images/6.png"],
                [f"{this_file_path}/example_images/7.png"],
                [f"{this_file_path}/example_images/8.png"],
                [f"{this_file_path}/example_images/9.png"],
            ],
            inputs=sketchpad
        )

    run_btn.click(run_inference, inputs=[sketchpad], outputs=[result_output], show_progress=False)

# This is required to use gradio through
# google Colab
interface.launch(share=True)




