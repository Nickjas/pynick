from logging import log
import streamlit as st
from sdv.tabular import GaussianCopula,CopulaGAN,CTGAN,TVAE
from pycaret.regression import predict_model


import plotly.graph_objects as go
import numpy as np
#=============================================================
def convert_str_to_list(text:str):
    """convert a text to list of int

    Args:
        text (str): the text to convert
    """
    if ',' in text:
        converted_text = [int(i) for i in text.split(',')]
    else:
        converted_text = int(text)
    return converted_text
#===========================================
def get_plotly_act_vs_predict(estimator,X_train,X_test,y_train,y_test):
    """draw the actual vs predction for regression plot

    Args:
        estimator: (object) trained pycaret ml model
        y_train ((pd.DataFrame, np.ndarray)): y training data
        y_train_pred ((pd.DataFrame, np.ndarray)): prediction on y training data
        y_test ((pd.DataFrame, np.ndarray)): y testing data
        y_test_pred ((pd.DataFrame, np.ndarray)): prediction on y testing data

    Returns:
        str: plotly figure object
    """

    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(y_train),y=np.array(y_train_pred),mode='markers',name="Train"))
    fig.add_trace(go.Scatter(x=np.array(y_test), y=np.array(y_test_pred),mode='markers',name="Test"))

    fig.add_shape(type="line",
        x0=0, y0=0, x1=1, y1=1,xref="paper",yref="paper",name="Identity",
        line=dict(
            color="black",width=2,dash="dot"))
    fig.update_layout(
        title={
            'text':"Actual Value vs. Predicted Value",
            'xanchor':'center',
            'yanchor': 'top',
            'x': 0.5},
        xaxis_title="Actual",
        yaxis_title="Predicted",
        margin=dict(l=40, r=40, t=40, b=40),
        width=1000)
    
    return fig


def gauge_plot(original_value,optimal_value, lower_bound, 
               upper_bound, min_value, max_value):
    """plot the gauge plot for backwards Analysis regression problem

    Args:
        original_value (float or int): the original Y value to optimize
        optimal_value (float or int): the optimal value found in generated data
        lower_bound (float or int): the lower bound value to optimize 
        upper_bound (float or int): the upper bound value to optimize
        min_value (float or int): the minimum value of target column
        max_value (float or int): the maximum value of target column    
    Returns:
        [object]: plotly gauge object to show
    """
    if original_value > optimal_value:
        delta = {'reference': original_value, 'increasing': {'color': "RebeccaPurple"}}
    else:
        delta = {'reference': original_value, 'decreasing': {'color': "RebeccaPurple"}}   
    
    
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = optimal_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Optimal", 'font': {'size': 24}},
        
        delta = delta,
        gauge = {
            'axis': {'range': [min_value, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                # {'range': [min_value, mean_value], 'color': 'cyan'},royalblue
                {'range': [lower_bound, upper_bound], 'color': 'cyan'}]}))
            # 'threshold': {
            #     'line': {'color': "red", 'width': 4},
            #     'thickness': 0.75,
            #     'value': max_value-reference_value}}))
    fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
    return fig


def find_top_5_nearest(array, value):
    """Find the top 5 closest neighbors of given optimal value

    Args:
        array (np.array): the generated data with prediction
        value (int or float): optimal value to find

    Returns:
        list: the top 5 indices of suggested value
    """
    array = np.asarray(array)
    diff = np.abs(array - value)
    indices = np.argsort(diff)
    return indices
    #====================================================================
def write(state):
    if state.trained_model is not None:
        
        X_before_preprocess = state.X_before_preprocess
        target_name = state.y_before_preprocess
        df_X = X_before_preprocess.drop(target_name,axis=1)
        trained_model = state.trained_model
        min_value = X_before_preprocess[target_name].min()
        max_value = X_before_preprocess[target_name].max()
        mean_value = X_before_preprocess[target_name].mean()
        original_value = optimal_value = mean_value 

        st.header("Knowledge Generation and Backward Analysis.")
        with st.beta_expander("Knowledge Generation"):
            st.markdown('<p style="color:#1386fc">Please Select a Method to Generate Data.</p>',unsafe_allow_html=True)     
            sdv_method = st.selectbox('Method to Generate Data', options=['GaussianCopula','CTGAN','CopulaGAN','TVAE'])
            sample = st.number_input('How Many Samples of Data to Generate?', min_value =1, value=df_X.shape[0],key=1)

            if sdv_method == 'GaussianCopula':
                model = GaussianCopula()
            else:
                is_tune = st.checkbox("Do You Want to Tune Hyperparameters?", value=False)
                if sdv_method == 'CopulaGAN' or sdv_method == 'CTGAN':
                    epochs = 300
                    batch_size=500
                    log_frequency=True
                    embedding_dim=128
                    generator_dim= (256,256)
                    discriminator_dim = (256,256)
                    generator_lr=0.0002
                    generator_decay=1e-6
                    discriminator_lr = 0.0002
                    discriminator_decay=1e-6
                    discriminator_steps = 1
                    
                    if is_tune:
                        epochs = st.number_input("Number of Training Epochs (int)", min_value=1, value=300,key=1)
                        batch_size = st.number_input("Number of Data Samples to Process, should be a multiple of 10 (int)", min_value =1, value=500,key=1)
                        log_frequency = st.checkbox('Whether to Use Log Frequency', value=True)
                        embedding_dim = st.number_input("Size of the Random Sample Passed to the Generator (int)", min_value=1, value=128,key=1)
                        generator_dim  = st.text_input("Size of the Generator Residual Layer (int)", value="256,256")
                        discriminator_dim = st.text_input("Size of the Discriminator Residual Layer (int)", value="256,256")
                        generator_lr = st.number_input("Learning Rate for the Generator", min_value=0.0, value=0.0002, format="%e")
                        generator_decay  = st.number_input("Generator Weight Decay for the Adam Optimizer", min_value=0.0, value=1e-6, format="%e")
                        discriminator_lr   = st.number_input("Learning Rate for the Discriminator", min_value=0.0, value=0.0002, format="%e")
                        discriminator_decay   = st.number_input("Discriminator  Weight Decay for the Adam Optimizer", min_value=0.0, value=1e-6, format="%e")
                        discriminator_steps  = st.number_input("Number of Discriminator Updates to do for Each Generator Update (int)", min_value=1, value=1)
                        
                        generator_dim = convert_str_to_list(generator_dim)
                        discriminator_dim = convert_str_to_list(discriminator_dim)
                    if sdv_method == 'CopulaGAN':
                        model = CopulaGAN(epochs=epochs, batch_size=batch_size,log_frequency=log_frequency,
                                        embedding_dim=embedding_dim,generator_dim=generator_dim,discriminator_dim=discriminator_dim,
                                        generator_lr=generator_lr,generator_decay=generator_decay,
                                        discriminator_lr=discriminator_lr,discriminator_decay=discriminator_decay,
                                        discriminator_steps=discriminator_steps)
                    if sdv_method == 'CTGAN':
                        model = CTGAN(epochs=epochs, batch_size=batch_size,log_frequency=log_frequency,
                                        embedding_dim=embedding_dim,generator_dim=generator_dim,discriminator_dim=discriminator_dim,
                                        generator_lr=generator_lr,generator_decay=generator_decay,
                                        discriminator_lr=discriminator_lr,discriminator_decay=discriminator_decay,
                                        discriminator_steps=discriminator_steps)
                else:
                    compress_dims =decompress_dims=(128,128)
                    epochs=300
                    batch_size=500
                    embedding_dim=128
                    l2_scale=1e-5
                    if is_tune:
                        epochs = st.number_input("Number of Training Epochs (int)", min_value=1, value=300,key=2)
                        batch_size = st.number_input("Number of Data Samples to Process, should be a multiple of 10 (int)", min_value =1, value=500,key=2)
                        embedding_dim = st.number_input("Size of the Random Sample Passed to the Generator (int)", min_value=1, value=128,key=2)
                        compress_dims  = st.text_input("Size of Each Hidden Layer in the Encoder (int)", value="128,128")
                        decompress_dims  = st.text_input("Size of Each Hidden Layer in the Decoder (int)", value="128,128")
                        l2_scale  = st.number_input("Regularization term", min_value=0.0, value=1e-5, format="%e")
                        
                        compress_dims = convert_str_to_list(compress_dims)
                        decompress_dims = convert_str_to_list(decompress_dims)
                    model = TVAE(embedding_dim=embedding_dim, compress_dims=compress_dims, decompress_dims=decompress_dims, 
                                 l2scale=l2_scale, batch_size=batch_size, epochs=epochs)

            button_generate = st.button("Generate")
            if button_generate:
                with st.spinner("Generating..."):
                    model.fit(df_X)
                    new_data = model.sample(sample)
                    new_data_prediction = predict_model(trained_model,new_data)
                    st.write(new_data_prediction)
                    state.new_data_prediction = new_data_prediction
                
        st.markdown("---")
        with st.beta_expander("Backward Analysis"):
            col1, col2 = st.beta_columns(2)
            with col1:
                st.subheader("Please Select a Index for Data to Optimize")
                index = st.number_input("Index of Data", min_value=0, value=0,max_value=df_X.shape[0]-1,key=1)
                st.write(X_before_preprocess.iloc[index])
                original_value = X_before_preprocess.iloc[index].loc[target_name]
                # st.write(original_value)
            with col2:
                st.subheader("Optimize")
                lower_bound = st.number_input("The Lower Bound Value to Optimize",value=min_value)
                upper_bound = st.number_input("The Upper Bound Value to Optimize",value=max_value)
                button_optimize = st.button("Optimizer")
                if button_optimize:
                    if state.new_data_prediction is not None:
                        new_prediction = state.new_data_prediction['Label']
                        indices = find_top_5_nearest(new_prediction, original_value)
                        optimal_value = new_prediction[indices[0]]
                        state.suggest_indices = indices
                        state.optimal_value = optimal_value
                    else:
                        st.error("Please Generate New Data first!")
                
        with st.beta_container():
            state.optimal_value = state.optimal_value if state.optimal_value is not None else 0
            fig = gauge_plot(original_value,state.optimal_value,lower_bound,
                             upper_bound,min_value,max_value)
            st.plotly_chart(fig)
            button_suggest = st.button("Show the Top 5 Suggestions")
            if button_suggest:
                suggestion = state.new_data_prediction.iloc[state.suggest_indices[:5]]
                st.table(suggestion)
    else:
        st.error("Please Train a Model first!")
        
