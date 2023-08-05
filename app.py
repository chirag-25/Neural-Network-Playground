import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)

# function to generate toy datasets
def generate_dataset(dataset_type='circle'):
    if dataset_type == 'circle':
        X = np.random.randn(200, 2)
        y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1).astype(int)
    
    elif dataset_type == 'spiral':
        t1 = np.linspace(0, 2*np.pi, 100)
        X1 = np.vstack([0.5 * t1 * np.cos(t1), 0.5 * t1 * np.sin(t1)]).T
        X1 += np.random.randn(100, 2) * 0.2
        y1 = np.zeros(100, dtype=int)

        t2 = np.linspace(0, 2*np.pi, 100)
        X2 = np.vstack([0.5 * t2 * np.cos(t2 + np.pi), 0.5 * t2 * np.sin(t2 + np.pi)]).T
        X2 += np.random.randn(100, 2) * 0.2
        y2 = np.ones(100, dtype=int)

        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))
    
    elif dataset_type == 'exclusive_or':
        X = np.random.randn(200, 2)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

    return X, y

    
# function to handle the different basis function
def apply_basis(X, include_basis=False, basis_type=["X1", "X2"]):
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    if include_basis:
        basis_functions = []
        if "X1" in basis_type:
            basis_functions.append(X1)
        if "X2" in basis_type:
            basis_functions.append(X2)
        if "X1^2" in basis_type:
            basis_functions.append(X1**2)
        if "X2^2" in basis_type:
            basis_functions.append(X2**2)
        if "X1X2" in basis_type:
            basis_functions.append(X1 * X2)
        if "sin(X1)" in basis_type:
            basis_functions.append(np.sin(X1))
        if "sin(X2)" in basis_type:
            basis_functions.append(np.sin(X2))
        if "gaussian X1" in basis_type:
            basis_functions.append(np.exp(-0.5 * (X1 - np.mean(X1))**2))
        if "gaussian X2" in basis_type:
            basis_functions.append(np.exp(-0.5 * (X2 - np.mean(X2))**2))
        
        X = np.column_stack(basis_functions)
        
    return X


# network creation
def create_model(num_features, num_hidden_layers, neurons_each_layer, learning_rate, dropout_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(num_features,)))
    # hidden layers
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_each_layer[i], activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))  # Add dropout layer
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# streamlit app
def main():
    st.title("Neural Network Contour Plot App")

    with st.sidebar:
        # setting different parameters
        dataset_type = st.selectbox("Select Dataset", ['circle', 'spiral', 'exclusive_or'])
        include_basis = st.checkbox("Include Basis Functions", value=False)
        if include_basis:
            basis_type = st.multiselect(
                'Select the inputs',
                ['X1', 'X2', 'X1^2', 'X2^2', 'X1X2', 'sin(X1)', 'sin(X2)', 'gaussian X1', 'gaussian X2'],
                ['X1', 'X2'])
            if(basis_type == []):
                st.warning('Cannot leave this empty', icon="⚠️")
                st.stop()
        else:
            basis_type = ["X1", "X2"]
        learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        epochs = st.slider("Number of Epochs", min_value=10, max_value=1000, value=50, step=10)
        dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
        include_MC_dropout = st.checkbox("Apply Monte Carlo Dropout", value=False)
        
        # hidden layers
        num_hidden_layers = st.slider("Number of Hidden Layers", min_value=1, max_value=10, value=2, step=1)
        neurons_each_layer = [2] * num_hidden_layers
        if(num_hidden_layers >=1):
            for i in range(num_hidden_layers):
                s = f"Layer {i+1}"
                neurons_each_layer[i] =  st.number_input(s, min_value=1, max_value=10, value = 3, step=1)
    
    
    # # dataset generation
    # X, y = generate_dataset(dataset_type)
    # X = apply_basis(X, include_basis, basis_type)
    # num_features= X.shape[1]
    # # neural network
    # model = create_model(num_features, num_hidden_layers,neurons_each_layer, learning_rate, dropout_rate)
    # history = model.fit(X, y, epochs=epochs, verbose=0)
    
    # dataset generation
    X, y = generate_dataset(dataset_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = apply_basis(X_train, include_basis, basis_type)
    X_test = apply_basis(X_test, include_basis, basis_type)
    num_features= X_train.shape[1]
    
    # neural network
    model = create_model(num_features, num_hidden_layers,neurons_each_layer, learning_rate, dropout_rate)
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0)
    
    y_pred = model.predict(X_test) # y_pred is in probabilities
    # Calculating accuracy
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred_binary == y_test)
    # st.write(f"Accuracy: {accuracy}")
    
    # plotting
    xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points = apply_basis(grid_points, include_basis, basis_type)
    probabilities = model.predict(grid_points).reshape(xx.shape)
    
    fig, ax = plt.subplots()
    contour = ax.contourf(xx, yy, probabilities, levels=20, cmap='RdYlBu', alpha=0.6)
    cbar = plt.colorbar(contour)
    cbar.ax.set_ylabel('Probability', rotation=270, labelpad=15)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    st.pyplot(fig)
    
    if include_MC_dropout:
        
        y_pred_mc = np.stack([model(X_test,training=True)
                    for sample in range(100)]) # y_pred is in probabilities
        y_pred_mc_mean = y_pred_mc.mean(axis=0)
        y_pred_binary_mc = (y_pred_mc_mean > 0.5).astype(int)
        accuracy_mc = np.mean(y_pred_binary_mc == y_test)

        y_prob_mc = np.stack([model(grid_points,training=True)
                    for sample in range(100)])
        y_prob_mc_mean = y_prob_mc.mean(axis=0).reshape(xx.shape)
        y_prob_mc_std = y_prob_mc.std(axis=0).reshape(xx.shape)
        
        st.header("With Monte Carlo Dropout")
        # st.write(f"Accuracy: {accuracy_mc}")
        fig1, ax = plt.subplots()
        contour = ax.contourf(xx, yy, y_prob_mc_mean, levels=20, cmap='RdYlBu', alpha=0.6)
        cbar = plt.colorbar(contour)
        cbar.ax.set_ylabel('Probability', rotation=270, labelpad=15)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
        st.pyplot(fig1)
    
    

if __name__ == "__main__":
    main()
