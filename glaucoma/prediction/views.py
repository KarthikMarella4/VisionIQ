from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import numpy as np
from PIL import Image
import os

# Try to import tflite_runtime, fallback to tensorflow.lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        try:
            import ai_edge_litert.interpreter as tflite
        except ImportError:
            raise ImportError("Neither tflite_runtime, tensorflow, nor ai_edge_litert is installed.")

def home(request):
    return render(request, 'index.html')

def preprocess_input_vgg19(x):
    # VGG19 preprocessing: RGB -> BGR, then zero-center
    # 'RGB'->'BGR'
    x = x[..., ::-1]
    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x

def prediction(request):
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtracting max value for numerical stability
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    if request.method == 'POST' and request.FILES['image']:
        try:
            print("DEBUG: Request received. Processing image...", flush=True)
            image_file = request.FILES['image']

            # Save temp file
            with open('temp_image.jpg', 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)
            print("DEBUG: Image saved to temp_image.jpg", flush=True)

            # TFLite Inference setup
            # Construct absolute path to ensure we find the file regardless of CWD
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # glaucama/ folder
            lite_model_path = os.path.join(base_dir, 'model.tflite')
            
            print(f"DEBUG: Calculated model path: {lite_model_path}", flush=True)

            # Check if model exists, if not, try to reconstruct from chunks
            if not os.path.exists(lite_model_path):
                 print(f"DEBUG: {lite_model_path} not found. Checking for chunks...", flush=True)
                 chunk_0 = lite_model_path + ".000"
                 if os.path.exists(chunk_0):
                     print("DEBUG: Found chunks. Reassembling model...", flush=True)
                     with open(lite_model_path, 'wb') as output_file:
                         part_num = 0
                         while True:
                             chunk_path = f"{lite_model_path}.{part_num:03d}"
                             if not os.path.exists(chunk_path):
                                 break
                             print(f"DEBUG: Merging chunk {chunk_path}", flush=True)
                             with open(chunk_path, 'rb') as input_chunk:
                                 output_file.write(input_chunk.read())
                             part_num += 1
                     print("DEBUG: Model reassembly complete.", flush=True)
                 else:
                     print(f"ERROR: Model chunks not found at {chunk_0}", flush=True)
            
            if not os.path.exists(lite_model_path):
                 # Fallback check current directory
                 if os.path.exists('model.tflite'):
                     lite_model_path = 'model.tflite'
                     print(f"DEBUG: Found model in current dir, using relative path: {lite_model_path}", flush=True)
                 else:
                     raise FileNotFoundError(f"Model file not found at {lite_model_path} or current dir")
            
            # Check file size to verify LFS download
            file_size_mb = os.path.getsize(lite_model_path) / (1024 * 1024)
            print(f"DEBUG: Model file size: {file_size_mb:.2f} MB", flush=True)
            
            if file_size_mb < 1:
                print("WARNING: Model file is too small! It might be a Git LFS pointer.", flush=True)
            
            # Helper to load interpreter depending on the imported library structure
            if hasattr(tflite, 'Interpreter'):
                interpreter = tflite.Interpreter(model_path=lite_model_path)
            else:
                interpreter = tflite.Interpreter(model_path=lite_model_path)
            
            print("DEBUG: Interpreter created. Allocating tensors...", flush=True)
            interpreter.allocate_tensors()
            print("DEBUG: Tensors allocated.", flush=True)

            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Load and preprocess image using PIL (removes keras dependency)
            print("DEBUG: Loading image with PIL...", flush=True)
            img = Image.open('temp_image.jpg').resize((256, 256))
            img_array = np.array(img).astype('float32')
            
            # Apply VGG19 preprocessing manually
            img_array = preprocess_input_vgg19(img_array)
            
            # Reshape for model input
            img_array = img_array.reshape(1, 256, 256, 3)
            print(f"DEBUG: Image preprocessed. Shape: {img_array.shape}", flush=True)

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], img_array)

            # Run inference
            print("DEBUG: Invoking interpreter (Running inference)...", flush=True)
            interpreter.invoke()
            print("DEBUG: Inference complete.", flush=True)

            # Get output tensor
            predict = interpreter.get_tensor(output_details[0]['index'])
            print(f"DEBUG: Prediction result raw: {predict}", flush=True)

            normalized_output_exp = softmax(predict)
            percentages_exp = normalized_output_exp * 100
            print(f"DEBUG: Percentages: {percentages_exp}", flush=True)
            
            result = np.argmax(predict)
            output = ''

            if result == 0:
                if percentages_exp[0][0] > 70:
                    output = "Severe Stage"
                else:
                    output = "Initial Stage"
            else:
                output = "Normal Eye"

            prediction = output
            print(f"DEBUG: Final prediction: {prediction}", flush=True)

        except Exception as e:
            print(f"ERROR: Exception during prediction: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)

        finally:
            # Clean up temp file
            if os.path.exists('temp_image.jpg'):
                os.remove('temp_image.jpg')
                print("DEBUG: Cleaned up temp file.", flush=True)

        return JsonResponse({'prediction': prediction})

    return HttpResponse("hi")

