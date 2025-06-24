import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size # 7
        self.embed_dim = embed_dim # 128
        self.num_patches = (image_size // patch_size) ** 2 # 16
        
        # Define layers and parameters
        self.patch_embedding = nn.Linear(patch_size * patch_size, embed_dim) # 49 -> 128
        self.position_embedding = nn.Parameter(torch.randn(self.num_patches, embed_dim)) # 16, 128
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # 1, 1, 128
    
    def forward(self, x):

        # x can be (1, 1, 28, 28) or (1, 1, 40, 40) where batch_size is dynamic (batch_size, channels, image_w, image_h)
        # 1. Patch tokenization
        # x = x.squeeze(1) # (1, 1, 28, 28) -> (1, 28, 28) drops the channel dimension
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # (1, 1, 7, 7, 4, 4)

        patches_flat = patches.reshape(x.shape[0], self.num_patches, self.patch_size * self.patch_size) # (1, 1, 16, 49)
        # print(f'{patches_flat.shape} - patches_flat')
        
        # 2. Linear projection (use class attribute)
        embedded_patches = self.patch_embedding(patches_flat) # (1, 1, 16, 128)
        
        # 3. Add positional embeddings (use class attribute)
        embedded_patches_with_pos = embedded_patches + self.position_embedding  # Skip class token position # (1, 1, 16, 128)
        
        # 4. Add class token (use class attribute) - optional
        embedded_patches_with_class = torch.cat([self.class_token.expand(x.shape[0], -1, -1), embedded_patches_with_pos], dim=1) # (1, 1, 16, 128)
        
        return embedded_patches_with_class

class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        
        # Define the linear layers (same as your function)
        self.W_q = nn.Linear(embed_dim, embed_dim) 
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)


    def forward(self, residual_stream):
        # Print weight shapes (same as your function)
        # print(self.W_q.weight.shape, self.W_k.weight.shape, self.W_v.weight.shape, self.W_o.weight.shape)

        # Compute Q, K, V (exactly the same logic)
        K = residual_stream @ self.W_k.weight.T  # (batch_size, 1, 16, 128)
        Q = residual_stream @ self.W_q.weight.T  # (batch_size, 1, 16, 128)
        V = residual_stream @ self.W_v.weight.T  # (batch_size, 1, 16, 128)
        
        # Compute attention scores (exactly the same logic)
        A = Q @ K.transpose(-1, -2)  # (batch_size, 1, 16, 16)
        A = A / (self.embed_dim ** 0.5)  # scale by sqrt(d_k) to avoid large values
        A = torch.softmax(A, dim=-1)  # Convert to probabilities
        
        # Apply attention (exactly the same logic)
        H = A @ V  # (batch_size, 1, 16, 128) # Raw attention output
        H = self.W_o(H)
        # H = H + self.W_o(H)   # (batch_size, 1, 16, 128) # Output of the attention layer


        # Uncomment these if you want the same debug prints
        # print(f'{K.shape} - K')
        # print(f'{Q.shape} - Q')
        # print(f'{V.shape} - V')
        # print(f'{A.shape} - A')
        # print(f'{H.shape} - H')
        
        return H

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super(MLP, self).__init__()
        self.embed_dim = embed_dim
        
        # Define the linear layers (same as your function)
        self.LayerNorm1 = nn.LayerNorm(embed_dim)
        self.mlp_up = nn.Linear(embed_dim, 4*embed_dim)
        self.mlp_down = nn.Linear(4*embed_dim, embed_dim)
        self.LayerNorm2 = nn.LayerNorm(embed_dim)

    def forward(self, H):

        # MLP - optional
        residual_stream = self.LayerNorm1(H)
        residual_stream_mlp_up = self.mlp_up(residual_stream)
        residual_stream_mlp_up = torch.relu(residual_stream_mlp_up)
        residual_stream = residual_stream + self.mlp_down(residual_stream_mlp_up) # residual connection
        residual_stream = self.LayerNorm2(residual_stream)

        return residual_stream

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Define the linear classifier (same as your function)
        self.Linear_classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, H):
        # Extract only the class token (first token) - same logic
        class_token_output = H[:, 0, :]  # Shape: (batch_size, 128)
        # print(f"Class token shape: {class_token_output.shape}")

        # Apply classifier only to class token - same logic
        logits = self.Linear_classifier(class_token_output)  # Shape: (batch_size, 10)
        # print(f"Logits shape: {logits.shape}")

        # Get prediction - same logic
        probability_dist = torch.softmax(logits, dim=1)
        # print(f"Probability distribution shape: {probability_dist.shape}")
        # print(f"Probability distribution: {probability_dist}")
        # predicted_class = torch.argmax(probability_dist, dim=1)
        # print(f"Predicted class: {predicted_class}")
        
        return logits

class Transformer1(nn.Module):
    """ This is a simple transformer model with 3 encoder layers and a classification head"""
    def __init__(self, image_size, patch_size, embed_dim, num_layers, num_classes):
        super(Transformer1, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim)
        
        # Create multiple encoder layers
        self.encoder_layers = nn.ModuleList([
            Encoder(embed_dim=embed_dim) for _ in range(num_layers)
        ])
        self.classification_head = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        
        # Pass through each encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        x = self.classification_head(x)
        return x


class Transformer2(nn.Module):
    """ This is a simple transformer model with 3 encoder layers, an MLP and a classification head"""
    def __init__(self, image_size, patch_size, embed_dim, num_layers, num_classes):
        super(Transformer2, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim)
        
        # Create multiple encoder layers
        self.encoder_layers = nn.ModuleList([
            Encoder(embed_dim=embed_dim) for _ in range(num_layers)
        ])
        self.mlp = MLP(embed_dim=embed_dim)
        self.classification_head = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        
        # Pass through each encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        x = self.mlp(x)
        
        x = self.classification_head(x)
        return x



def test_patch_embedding():
    image_size = 28
    channels = 1
    patch_size = 7
    embed_dim = 128
    batch_size = 1

    # Create the module once
    patch_embedding = PatchEmbedding(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim)

    # Use it multiple times with the same learned weights
    image1 = torch.randn(batch_size, channels, image_size, image_size)
    image2 = torch.randn(batch_size, channels, image_size, image_size)


    patch_embedded_image1 = patch_embedding(image1)  # Uses learned weights
    patch_embedded_image1 = patch_embedding(image2)  # Uses same learned weights

    print(f'{patch_embedded_image1.shape} - patch_embedded_image1')
    print(f'{patch_embedded_image1.shape} - patch_embedded_image1')

    return patch_embedded_image1

def test_encoder(patch_embedded_image):
    # Create encoder once
    encoder = Encoder(embed_dim=128)

    layers = 3
    # Use it multiple times (same weights)
    H = patch_embedded_image
    for i in range(layers):
        H = encoder(H)  # Uses the same learned weights each time

    print(f'{H} - H')
    return H

def test_classification_head(H):
    # Create classification head
    classification_head = ClassificationHead(embed_dim=128, num_classes=10)

    # Use it (same as your function)
    predicted_class = torch.argmax(classification_head(H), dim=1)  # Uses the same learned weights
    print(f'{predicted_class} - predicted_class')
    return predicted_class

def test_transformer():
    image_size = 28
    channels = 1
    patch_size = 7
    embed_dim = 128
    batch_size = 5
    num_encoder_layers = 3
    num_classes = 10

    # Create the complete model with 3 layers
    model = Transformer1(
        image_size=image_size, 
        patch_size=patch_size, 
        embed_dim=embed_dim, 
        num_layers=num_encoder_layers, 
        num_classes=num_classes
    )

    # Single forward pass handles all layers
    images = torch.randn(batch_size, channels, image_size, image_size)  # batch of 4 images
    predictions = model(images)  # Automatically goes through all 3 layers
    print(f'{predictions} - predictions')


if __name__ == "__main__":
    
    # test_classification_head(test_encoder(test_patch_embedding()))
    
    test_transformer()