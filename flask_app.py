from flask import Flask, request, jsonify
import torch
from functools import partial
from models_vit import VisionTransformer
import os
from torch.cuda.amp import autocast
import torch.nn.functional as F

app = Flask(__name__)

def infer(model, images, device, num_class):
    """
    Perform inference on a batch of images using the model.

    Args:
        model (torch.nn.Module): The trained model.
        images (torch.Tensor): Batch of input images.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        num_class (int): Number of classes for one-hot encoding.

    Returns:
        torch.Tensor: Predicted labels.
        torch.Tensor: Softmax probabilities.
    """
    images = images.to(device, non_blocking=True)
    
    with autocast():
        output = model(images)
    
    output_ = F.softmax(output, dim=1)
    output_label = output_.argmax(dim=1)
    output_onehot = F.one_hot(output_label.to(torch.int64), num_classes=num_class)
    
    return output_label, output_

def load_model(checkpoint_path, num_classes=3):  # Changed default to 3 based on checkpoint
    # First load the checkpoint to inspect its structure
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Determine the number of classes from the checkpoint if possible
    if 'head.weight' in state_dict:
        num_classes = state_dict['head.weight'].shape[0]
    elif 'head.bias' in state_dict:
        num_classes = state_dict['head.bias'].shape[0]
    
    print(f"Initializing model with {num_classes} classes")
    
    # Initialize model with parameters matching the checkpoint
    model = VisionTransformer(
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        global_pool=True,
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,  # Increased depth based on the error showing blocks up to 23
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )
    
    # Load state dict, ignoring any size mismatches
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model with message: {msg}")
    
    # If there were missing keys, try to handle them
    if msg.missing_keys:
        print(f"Warning: Missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"Warning: Unexpected keys: {msg.unexpected_keys}")
    
    model.eval()
    return model

# Load the pre-trained models
models = {
    'DR': load_model('models/MESSIDOR2/checkpoint-best.pth'),
    'AMD': load_model('models/APTOS/checkpoint-best.pth'),
    'Glaucoma': load_model('models/glaucoma/checkpoint-best.pth')
}

@app.route('/health_assessment', methods=['POST'])
def health_assessment():
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'actual_age' not in data:
            return jsonify({'error': 'Missing required fields: image and actual_age are required'}), 400
        
        # Decode base64 image
        import base64
        from io import BytesIO
        from PIL import Image
        import numpy as np
        
        try:
            # Decode base64 string to bytes
            img_data = base64.b64decode(data['image'])
            # Convert bytes to PIL Image
            img = Image.open(BytesIO(img_data)).convert('RGB')
            
            # Resize and normalize the image
            img = img.resize((224, 224))  # Resize to expected input size
            img_array = np.array(img).astype(np.float32) / 255.0  # Convert to float and normalize
            
            # Convert to PyTorch tensor and add batch dimension
            images = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW
            
            # Normalize with ImageNet mean and std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = (images - mean) / std
            
            results = {}
            for condition, model in models.items():
                # Determine number of classes based on the model's output layer
                num_classes = model.head.out_features if hasattr(model, 'head') else 5
                output_label, output_prob = infer(model, images, torch.device('cpu'), num_class=num_classes)
                
                results[condition] = {
                    'score': output_label.item(),
                    'confidence': output_prob.max().item(),
                    'probabilities': output_prob.squeeze().tolist()
                }
            
            # Calculate biological age (placeholder)
            results['Biological age'] = int(data['actual_age'])
            results['report_image'] = data['image']
            
            return jsonify(results)
            
        except Exception as e:
            import traceback
            return jsonify({
                'error': 'Error processing image',
                'details': str(e),
                'traceback': traceback.format_exc()
            }), 400
            
    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)