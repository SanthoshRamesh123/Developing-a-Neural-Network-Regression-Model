# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1820" height="1017" alt="image" src="https://github.com/user-attachments/assets/f9a3a2e5-462a-49ac-824f-a216c6937fdb" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM
```
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()


        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
### Name:
SANTHOSH R
### Register Number:
212224230249

### Dataset Information
<img width="252" height="287" alt="image" src="https://github.com/user-attachments/assets/51f57d88-488a-4a41-ae4c-ebae7a47cd26" />

### OUTPUT
<img width="362" height="230" alt="image" src="https://github.com/user-attachments/assets/39c598a6-a62b-481d-acc2-7e6a96c9c6a3" />

### Training Loss Vs Iteration Plot
<img width="727" height="568" alt="image" src="https://github.com/user-attachments/assets/15eecd42-2fd1-43a5-913b-af3546f00c1d" />

### New Sample Data Prediction
<img width="422" height="37" alt="image" src="https://github.com/user-attachments/assets/4b28ee62-8ae9-4692-bb9a-0bd084c6e627" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
