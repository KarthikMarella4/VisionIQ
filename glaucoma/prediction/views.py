from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import numpy as np
from PIL import Image
import os

# Try to import tflite_runtime, fallback to tensorflow.lite if not available
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("Neither tflite_runtime nor tensorflow is installed.")

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
        image_file = request.FILES['image']

        # Save temp file
        with open('temp_image.jpg', 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        try:
            # TFLite Inference setup
            lite_model_path = 'model.tflite'
            
            # Helper to load interpreter depending on the imported library structure
            if hasattr(tflite, 'Interpreter'):
                interpreter = tflite.Interpreter(model_path=lite_model_path)
            else:
                interpreter = tflite.Interpreter(model_path=lite_model_path)
                
            interpreter.allocate_tensors()

            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Load and preprocess image using PIL (removes keras dependency)
            img = Image.open('temp_image.jpg').resize((256, 256))
            img_array = np.array(img).astype('float32')
            
            # Apply VGG19 preprocessing manually
            img_array = preprocess_input_vgg19(img_array)
            
            # Reshape for model input
            img_array = img_array.reshape(1, 256, 256, 3)

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], img_array)

            # Run inference
            interpreter.invoke()

            # Get output tensor
            predict = interpreter.get_tensor(output_details[0]['index'])

            normalized_output_exp = softmax(predict)
            percentages_exp = normalized_output_exp * 100
            print(percentages_exp)
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

        finally:
            # Clean up temp file
            if os.path.exists('temp_image.jpg'):
                os.remove('temp_image.jpg')

        return JsonResponse({'prediction': prediction})

    return HttpResponse("hi")

