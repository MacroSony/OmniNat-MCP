from mcp.server.fastmcp import FastMCP
import base64
import io

import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Initialize the FastMCP server
mcp = FastMCP("omninat-mcp")

# Load model and processor globally
model_id = "hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"
model = timm.create_model(model_id, pretrained=True)
labels = model.pretrained_cfg.get('label_names', [])
model.eval()  # Set to evaluation mode

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

@mcp.tool()
def identify_species(image_data: str) -> str:
    """
    Analyzes an image and identifies the species using the iNaturalist dataset.
    
    Args:
        image_data: A base64-encoded string of the image, or a data URI format (e.g., data:image/jpeg;base64,...).
    """
    try:
        # Extract the base64 part if it's a data URI
        if image_data.startswith("data:"):
            base64_str = image_data.split("base64,")[-1]
        else:
            base64_str = image_data
            
        # Decode the base64 string
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)

        return f"Predicted class ID: {labels[top_catid.item()]} with confidence {top_prob.item():.4f}"
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def main():
    """Runs the MCP server over standard input/output (stdio)"""
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
