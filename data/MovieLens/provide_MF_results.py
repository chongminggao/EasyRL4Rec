import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# Load the Movielens ratings data
CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "data_raw")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data():
    ratings_df = pd.read_csv(os.path.join(DATAPATH, "ratings.dat"), delimiter='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')

    # Split the data into training and test sets (10% for test)
    train_df, test_df = train_test_split(ratings_df, test_size=0.1, random_state=42)
    print(train_df)
    print(test_df)
    train_df.to_csv(os.path.join(DATAPATH, "movielens-1m-test.csv"), index=False) # needs to change train and test
    test_df.to_csv(os.path.join(DATAPATH, "movielens-1m-train.csv"), index=False) # needs to change train and test
    return ratings_df, train_df, test_df

# Defining the Dataset class for PyTorch
class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users = torch.tensor(ratings['UserID'].values, dtype=torch.int64)
        self.movies = torch.tensor(ratings['MovieID'].values, dtype=torch.int64)
        self.ratings = torch.tensor(ratings['Rating'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# Defining the Matrix Factorization Model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, num_factors):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors, sparse=False)
        self.movie_factors = nn.Embedding(num_movies, num_factors, sparse=False)
    
    def forward(self, user, movie):
        return (self.user_factors(user) * self.movie_factors(movie)).sum(1)




# Training and Evaluation Function
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        for users, movies, ratings in tqdm(train_loader, desc=f'Epoch {epoch+1} - Training'):
            users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(users, movies)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)

        # Evaluation Phase
        model.eval()
        total_test_loss = 0
        total_test_mae = 0
        with torch.no_grad():
            for users, movies, ratings in tqdm(test_loader, desc=f'Epoch {epoch+1} - Evaluating'):
                users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
                predictions = model(users, movies)
                loss = criterion(predictions, ratings)
                total_test_loss += loss.item()
                total_test_mae += torch.mean(torch.abs(predictions - ratings)).item()

        average_test_loss = total_test_loss / len(test_loader)
        average_test_mae = total_test_mae / len(test_loader)

        print(f"Epoch {epoch+1}, Training Loss: {average_train_loss:.4f}, Test Loss (MSE): {average_test_loss:.4f}, Test MAE: {average_test_mae:.4f}")

def main(device=device):
    ratings_df, train_df, test_df = load_data()
    # Initializing the dataset and dataloaders
    train_dataset = MovieLensDataset(train_df)
    test_dataset = MovieLensDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model parameters
    num_users = ratings_df['UserID'].max() + 1
    num_movies = ratings_df['MovieID'].max() + 1
    num_factors = 6  # Number of latent factors

    # Model, loss function and optimizer
    model = MatrixFactorization(num_users, num_movies, num_factors).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Running the Training and Evaluation
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=80)

    # Save the trained model
    model_save_path = os.path.join(DATAPATH, 'trained_model.pth')
    torch.save(model.state_dict(), model_save_path)

    # Predict the entire rating matrix
    model.eval()  # Set the model to evaluation mode
    all_users = torch.tensor(range(num_users - 1), dtype=torch.int64).to(device) + 1
    all_movies = torch.tensor(range(num_movies - 1), dtype=torch.int64).to(device) + 1

    # Use torch.cartesian_prod to generate all user-movie pairs
    all_user_movie_pairs = torch.cartesian_prod(all_users, all_movies)

    # Predict ratings for all pairs
    with torch.no_grad():
        all_predictions = model(all_user_movie_pairs[:, 0], all_user_movie_pairs[:, 1])

    # Reshape the predictions to a matrix
    rating_matrix = all_predictions.view(num_users - 1, num_movies - 1).cpu().numpy()

    # Save the rating matrix to a file
    rating_matrix_save_path = os.path.join(DATAPATH, 'rating_matrix.csv')
    np.savetxt(rating_matrix_save_path, rating_matrix, delimiter=',')
    return rating_matrix
    
if __name__ == "__main__":
    main()