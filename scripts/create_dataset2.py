import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class CreditScoringGAN:
    def __init__(self, input_dim, categorical_dims, continuous_dims):
        self.input_dim = input_dim
        self.categorical_dims = categorical_dims
        self.continuous_dims = continuous_dims
        self.latent_dim = 100
        
        # Initialize networks
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
    def _build_generator(self):
        """Generator creates synthetic data from random noise"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, self.input_dim),
            nn.Tanh()  # Output normalized to [-1, 1]
        )
    
    def _build_discriminator(self):
        """Discriminator learns to distinguish real data from generated synthetic data"""
        return nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()  # Outputs probability: 1 = real data, 0 = fake data
        )
    
    def train(self, real_data, epochs=1000, batch_size=64):
        """
        Train the GAN:
        - Discriminator learns on REAL DATA vs SYNTHETIC DATA from generator
        - Generator learns to create synthetic data that fools the discriminator
        """
        # Convert real data to tensor
        real_dataset = torch.FloatTensor(real_data)
        dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for i, real_batch in enumerate(dataloader):
                current_batch_size = real_batch.size(0)
                
                # ===================
                # Train Discriminator
                # ===================
                self.d_optimizer.zero_grad()
                
                # 1. Train discriminator on REAL DATA
                real_labels = torch.ones(current_batch_size, 1)  # Label: 1 = real
                real_output = self.discriminator(real_batch)
                d_real_loss = self.criterion(real_output, real_labels)
                
                # 2. Train discriminator on SYNTHETIC DATA from generator
                noise = torch.randn(current_batch_size, self.latent_dim)
                synthetic_data = self.generator(noise)  # Generator creates fake data
                fake_labels = torch.zeros(current_batch_size, 1)  # Label: 0 = fake
                fake_output = self.discriminator(synthetic_data.detach())  # Don't backprop through generator
                d_fake_loss = self.criterion(fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                # ===============
                # Train Generator
                # ===============
                self.g_optimizer.zero_grad()
                
                # Generator tries to fool discriminator by making synthetic data look real
                noise = torch.randn(current_batch_size, self.latent_dim)
                synthetic_data = self.generator(noise)
                fake_output = self.discriminator(synthetic_data)  # Now backprop through generator
                
                # Generator wants discriminator to classify synthetic data as real (label = 1)
                g_loss = self.criterion(fake_output, real_labels)  # Use real_labels to fool discriminator
                g_loss.backward()
                self.g_optimizer.step()
            
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}]')
                print(f'  Discriminator Loss: {d_loss.item():.4f} (Real: {d_real_loss.item():.4f}, Fake: {d_fake_loss.item():.4f})')
                print(f'  Generator Loss: {g_loss.item():.4f}')
                print(f'  Discriminator accuracy on real data: {(real_output > 0.5).float().mean():.3f}')
                print(f'  Discriminator accuracy on fake data: {(fake_output <= 0.5).float().mean():.3f}')
                print('-' * 60)
    
    def generate_samples(self, n_samples):
        """
        Generate synthetic credit scoring data using trained generator
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.latent_dim)
            synthetic_data = self.generator(noise)
        self.generator.train()
        return synthetic_data.numpy()
    
    def discriminator_score(self, data):
        """
        Get discriminator's assessment of how 'real' the data looks
        Returns probability between 0-1 (1 = looks real, 0 = looks fake)
        """
        self.discriminator.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            scores = self.discriminator(data)
        self.discriminator.train()
        return scores.numpy()

class CreditDataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.categorical_columns = [
            'Employment_Sector', 'Salary_Frequency', 'Housing_Status',
            'Other_Income_Source', 'Household_Head', 'Comaker_Relationship',
            'Has_Community_Role', 'Paluwagan_Participation', 'Disaster_Preparedness'
        ]
        self.numerical_columns = [
            'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Years_at_Current_Address', 'Number_of_Dependents',
            'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff'
        ]
    
    def fit_transform(self, df):
        """
        Preprocess the REAL DATA for GAN training
        """
        processed_df = df.copy()
        
        # Handle categorical variables
        for col in self.categorical_columns:
            if col in processed_df.columns:
                encoder = LabelEncoder()
                processed_df[col] = encoder.fit_transform(processed_df[col].astype(str))
                self.encoders[col] = encoder
        
        # Handle numerical variables - normalize to [-1, 1] for tanh activation
        for col in self.numerical_columns:
            if col in processed_df.columns:
                scaler = StandardScaler()
                processed_df[col] = scaler.fit_transform(processed_df[[col]])
                self.scalers[col] = scaler
        
        return processed_df.values
    
    def inverse_transform(self, synthetic_data_array, original_columns):
        """
        Convert SYNTHETIC DATA back to original format
        """
        synthetic_df = pd.DataFrame(synthetic_data_array, columns=original_columns)
        
        # Reverse numerical scaling
        for col in self.numerical_columns:
            if col in synthetic_df.columns and col in self.scalers:
                synthetic_df[col] = self.scalers[col].inverse_transform(synthetic_df[[col]])
        
        # Reverse categorical encoding
        for col in self.categorical_columns:
            if col in synthetic_df.columns and col in self.encoders:
                # Round to nearest integer for categorical data
                synthetic_df[col] = np.round(synthetic_df[col]).astype(int)
                # Clip to valid range
                valid_range = range(len(self.encoders[col].classes_))
                synthetic_df[col] = np.clip(synthetic_df[col], min(valid_range), max(valid_range))
                # Inverse transform
                synthetic_df[col] = self.encoders[col].inverse_transform(synthetic_df[col])
        
        return synthetic_df

# Usage example
def train_credit_scoring_gan(real_data_path, sample_size=None, output_path="synthetic_credit_data.csv", n_synthetic=None):
    """
    Complete pipeline:
    1. Load REAL credit data (can be small sample)
    2. Train discriminator to recognize real data patterns
    3. Train generator to create synthetic data that looks real
    4. Save synthetic data to CSV file
    """
    # Load real data
    df = pd.read_csv(real_data_path)
    
    # Option to use only a small sample of real data
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Using {sample_size} samples from real data for training")
    
    print(f"Training with {len(df)} real data samples")
    
    # Preprocess data
    preprocessor = CreditDataPreprocessor()
    processed_real_data = preprocessor.fit_transform(df)
    
    # Initialize and train GAN
    input_dim = processed_real_data.shape[1]
    categorical_dims = len(preprocessor.categorical_columns)
    continuous_dims = len(preprocessor.numerical_columns)
    
    gan = CreditScoringGAN(input_dim, categorical_dims, continuous_dims)
    
    print("Starting GAN training...")
    print("Discriminator will learn from REAL data")
    print("Generator will learn to create SYNTHETIC data that fools discriminator")
    
    gan.train(processed_real_data, epochs=2000, batch_size=min(64, len(df)//2))
    
    # Generate synthetic data
    print("Generating synthetic data...")
    if n_synthetic is None:
        n_synthetic = len(df) * 5  # Generate 5x more synthetic data by default
    
    synthetic_data = gan.generate_samples(n_synthetic)
    
    # Convert back to original format
    synthetic_df = preprocessor.inverse_transform(synthetic_data, df.columns)
    
    # Save to CSV file
    synthetic_df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to: {output_path}")
    print(f"Generated {len(synthetic_df)} synthetic samples")
    
    return synthetic_df, gan, preprocessor

# Additional utility functions
def evaluate_gan_quality(real_df, synthetic_df, gan, preprocessor):
    """
    Evaluate how well the generator fools the discriminator
    """
    # Preprocess both datasets
    real_processed = preprocessor.fit_transform(real_df)
    synthetic_processed = preprocessor.fit_transform(synthetic_df)
    
    # Get discriminator scores
    real_scores = gan.discriminator_score(real_processed)
    synthetic_scores = gan.discriminator_score(synthetic_processed)
    
    print("GAN Quality Evaluation:")
    print(f"Real data - Average discriminator score: {real_scores.mean():.3f} (should be close to 1.0)")
    print(f"Synthetic data - Average discriminator score: {synthetic_scores.mean():.3f} (should be close to 1.0 if generator is good)")
    print(f"Discriminator can distinguish real from synthetic: {abs(real_scores.mean() - synthetic_scores.mean()):.3f} (lower is better)")

def validate_synthetic_data(real_df, synthetic_df):
    """
    Compare statistical properties of real vs synthetic data
    """
    import matplotlib.pyplot as plt
    
    # Statistical comparison
    print("Statistical Comparison:")
    print("Real Data Statistics:")
    print(real_df.describe())
    print("\nSynthetic Data Statistics:")
    print(synthetic_df.describe())
    
    # Distribution comparison for numerical columns
    numerical_cols = real_df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 0:
        fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(12, 4*len(numerical_cols)))
        if len(numerical_cols) == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numerical_cols):
            axes[i,0].hist(real_df[col].dropna(), bins=30, alpha=0.7, label='Real')
            axes[i,0].set_title(f'{col} - Real Data')
            
            axes[i,1].hist(synthetic_df[col].dropna(), bins=30, alpha=0.7, label='Synthetic', color='orange')
            axes[i,1].set_title(f'{col} - Synthetic Data')
        
        plt.tight_layout()
        plt.show()

# Example usage with small real data sample
if __name__ == "__main__":
    # Train GAN with small sample of real data and save synthetic data to CSV
    synthetic_df, trained_gan, preprocessor = train_credit_scoring_gan(
        real_data_path="credit_data.csv", 
        sample_size=1000,  # Use only 1000 real samples to train discriminator
        output_path="synthetic_credit_data.csv",  # Output CSV file
        n_synthetic=5000  # Generate 5000 synthetic samples
    )
    
    print(f"Generated {len(synthetic_df)} synthetic samples from {1000} real samples!")
    print("Synthetic data saved to synthetic_credit_data.csv")
    
    # Optional: Also save a comparison report
    real_df_full = pd.read_csv("credit_data.csv")
    if len(real_df_full) >= 1000:
        real_df = real_df_full.sample(n=1000, random_state=42)
    else:
        real_df = real_df_full.copy()
        print(f"Warning: Only {len(real_df)} real samples available for comparison.")
    
    # Create a comparison report and save to CSV
    comparison_stats = pd.DataFrame({
        'Real_Mean': real_df.select_dtypes(include=[np.number]).mean(),
        'Synthetic_Mean': synthetic_df.select_dtypes(include=[np.number]).mean(),
        'Real_Std': real_df.select_dtypes(include=[np.number]).std(),
        'Synthetic_Std': synthetic_df.select_dtypes(include=[np.number]).std()
    })
    comparison_stats.to_csv("real_vs_synthetic_comparison.csv")
    print("Statistical comparison saved to real_vs_synthetic_comparison.csv")

def save_synthetic_data_separately(gan, preprocessor, original_columns, n_samples=10000, output_path="synthetic_data.csv"):
    """
    Generate and save synthetic data to CSV after training
    """
    print(f"Generating {n_samples} synthetic samples...")
    synthetic_data = gan.generate_samples(n_samples)
    synthetic_df = preprocessor.inverse_transform(synthetic_data, original_columns)
    
    # Save to CSV
    synthetic_df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to: {output_path}")
    
    return synthetic_df