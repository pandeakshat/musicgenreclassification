#region ImportDependencies
from seaborn import regression
import streamlit as st
import librosa
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output
import glob
import imageio
import time
import IPython.display as ipd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time  # to calculate time taken for each genre calculation
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
import youtube_dl
import librosa  # to calculate features from the audio file
import librosa.display
from PIL import Image
import pathlib
import csv  # for loading the features into a file for future use
import warnings
import seaborn as sns
from scipy.stats import uniform, randint
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
from sklearn.linear_model import LogisticRegression
import sklearn.ensemble as ske
import eli5
from eli5.sklearn import PermutationImportance
from pprint import pprint
import random
import IPython
import librosa.display as lplt
import time
from matplotlib.ticker import AutoLocator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import wave
import IPython
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance
import catboost as cb
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import librosa
import os
import sys
import matplotlib
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import which
from PIL import Image
import nightcore as nc
import pyaudio


#endregion
# region Extras
matplotlib.use('Agg')
seed = 12
np.random.seed(seed)
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')
favicon = "music.png"
with Image.open(favicon) as image:
    width, height = image.size

st.set_page_config(page_title='Music Genre Classification',
                   page_icon=favicon, layout='wide', initial_sidebar_state='auto')
# endregion
# region Sidebar
st.sidebar.title("WebApp: \n Music Genre Classification using AI")

st.sidebar.header(" Applications:")
nav = st.sidebar.radio(
    ' ', ["Introduction", "Basic", "Advanced", "Conclusion"])
# endregion
# region Introduciton
if(nav == 'Introduction'):

    st.title("Introduction")
    st.header(" Title : Music Genre Classification using AI")
    st.write("This project proposes to create a machine learning model in the domain of music classification, a complex implementation of AI techniques to classify different music into genres by utilizing different features present in an audio file, pre-existing classified database of genre-specific songs and classifiers for analyzing. After classification and analyzing of the audio file, we perform conversion of genres, transformation on the audio files, a simple recommendation system based on the classification and music creation/generation based on our genre-model classification, and a front-end for interface. ")
    st.subheader("Applications :: Basic")
    st.write("    1. Database|Dataset")
    st.write("    2. Feature Extraction")
    st.write("    3. Model Training/Testing")
    st.write("    4. Genre Recognition")
    st.subheader("Applications :: Advanced")
    st.write("    1. Genre Conversion")
    st.write("    2. Genre Recommendation")
    st.write("    3. Genre Transformation")
    st.write("    4. Genre Creation")
    st.sidebar.header("Introduction")
    st.write(" ")
# endregion
# region Basic
if(nav == "Basic"):
    st.title("Basic Applications")
    st.sidebar.header("Basic Applications")
    options = st.sidebar.radio(
        "", ("Introduction", "Features", "Training/Testing", "Genre Recognition"))
    # region Definition

    def definitionnb():
        st.markdown('''Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, 
    			represented as vectors of feature values, where the class labels are drawn from some finite set. There is not a single algorithm
    		 	for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that 
    		 	the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a 
    		 	fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers
    		  	each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible
    		   	correlations between the color, roundness, and diameter features. ''')

    def definitionscd():
        st.markdown('''Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to fitting linear classifiers and regressors 
    			under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. SGD is merely an optimization 
    			technique and does not correspond to a specific family of machine learning models. It is only a way to train a model. Often, 
    			an instance of SGDClassifier or SGDRegressor will have an equivalent estimator in the scikit-learn API, potentially using a 
    			ifferent optimization technique. ''')

    def definitionknn():
        st.markdown('''In statistics, the k-nearest neighbors algorithm (k-NN) is a non-parametric classification method first developed by
    		 Evelyn Fix and Joseph Hodges in 1951, and later expanded by Thomas Cover.It is used for classification and regression. 
    		 In both cases, the input consists of the k closest training examples in data set. The output depends on whether k-NN is used for 
    		 classification or regression:  ''')

    def definitiondt():
        st.markdown('''A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences,
    		 including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains 
    		 conditional control statements.  ''')

    def definitionrf():
        st.markdown('''Random forests or random decision forests are an ensemble learning method for classification, regression and other
    		 tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the 
    		 classes (classification) or mean/average prediction (regression) of the individual trees.Random decision forests correct for decision
    		  trees' habit of overfitting to their training set.Random forests generally outperform decision trees, but their accuracy is lower 
    		  than gradient boosted trees. However, data characteristics can affect their performance. ''')

    def definitionsvm():
        st.markdown('''A support vector machine (SVM) is a supervised machine learning model that uses classification algorithms for
    		 two-group classification problems. After giving an SVM model sets of labeled training data for each category, they’re able to 
    		 categorize new text. ''')

    def definitionlr():
        st.markdown('''Logistic regression is named for the function used at the core of the method, the logistic function.
    			The logistic function, also called the sigmoid function was developed by statisticians to describe properties of population growth in ecology,
    			rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map
    			it into a value between 0 and 1, but never exactly at those limits.
    			1 / (1 + e^-value)
    			Where e is the base of the natural logarithms (Euler’s number or the EXP() function in your spreadsheet) and value is the actual numerical
    			value that you want to transform. ''')

    def definitionnn():
        st.markdown('''A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of 
    			artificial neurons or nodes.[1] Thus a neural network is either a biological neural network, made up of real biological neurons, 
    			or an artificial neural network, for solving artificial intelligence (AI) problems. The connections of the biological neuron are
    			 modeled as weights. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. 
    			 All inputs are modified by a weight and summed ''')

    def definitionxgb():
        st.markdown('''XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
    		 It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting 
    		 (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. ''')

    def definitionxgbrf():
        st.markdown('''XGBoost is normally used to train gradient-boosted decision trees and other gradient boosted models. Random forests 
    			use the same model representation and inference, as gradient-boosted decision trees, but a different training algorithm. 
    			One can use XGBoost to train a standalone random forest or use random forest as a base model for gradient boosting. 
    			Here we focus on training standalone random forest. ''')

    def definitioncat():
        st.markdown('''CatBoost is a machine learning method based on gradient boosting over decision trees.
    			Main advantages of CatBoost:
    			-Superior quality when compared with other GBDT libraries on many datasets.
    			-Best in class prediction speed.
    			-Support for both numerical and categorical features.
    			-Fast GPU and multi-GPU support for training out of the box.
    			-Visualization tools included.''')
    # endregion

    df = pd.read_csv(f"Dataset/data.csv")
    ef = df.label.value_counts().reset_index()
    df3 = df.shape
    df4 = df.size
    if(options == "Introduction"):
        st.write("Abstract")
        st.markdown(''' 
        Music, being one of the largest entertainment industries in the world, comes up with new songs being made every hour competing for the top billboard charts and to be listened by millions worldwide, having different genres and lyrics to impact the human mind in an uncountable number of ways. As the world is slowly shifting towards a more technological dependent era, where most models are slowly being shifted from calculative to a self-learning approach, as it gets popularized, we tend to see more models and projects that slowly adopt machine learning into different domains that are used by the people to see if they can differentiate between what was created by a human and its complex mind, with something created through a machine, without the user input, just data. Coming back to music, it affects the human brain in bizarre ways, not just our auditory senses, but music affects or physiological or psychological state according to research, and hence, given the different forms of music genres that exist, is it possible to implement the entirety of it, convert them into a technological model, and let the machine understand what music is, which comes naturally to human beings, can the machine, predict the genre of the song, make it sound like something else entirely, transform it into its own choices and finally create its own music and become an artist in itself. 
        ''')
        st.subheader("Applications :")
        st.write("Features :")
        st.write(" --> Create Dataset with Code Snippet")
        st.write(" --> Dataset and Audio Features")
        st.write("Training & Testing :")
        st.write(" --> Select Dataset")
        st.write(" --> Train under multiple ML algorithm with confusion matrix")
        st.write("Genre Recognition :")
        st.write(" --> Upload Audio | Youtube Url | Record Audio")
        st.write(" --> Recognize audio using ML and return predicted Genre.")

    if (options == 'Features'):
        choice = st.selectbox("Select choice", ("Create New Dataset[Tutorial]", "Dataset & Audio Features"))
        if (choice == "Dataset & Audio Features"):
            dataset_checkbox = st.selectbox(
                "Select or Upload", ("Select from existing", "Upload Dataset"))
            if (dataset_checkbox == "Upload Dataset"):
                Dataset = st.file_uploader("Upload the Dataset", type=['csv'])
                if Dataset is not None:
                    dataset = pd.read_csv(Dataset)
            elif (dataset_checkbox == "Select from existing"):
                select_dataset = st.selectbox(
                    "Select Dataset", ("1. Mean Extracted Dataset", "2. Mean and Variance Dataset", "3. GTZAN Dataset"))
                if (select_dataset == "1. Mean Extracted Dataset"):
                    dataset = pd.read_csv(f"Dataset/data.csv")
                elif (select_dataset == "2. Mean and Variance Dataset"):
                    dataset = pd.read_csv(f"Dataset/data2.csv")
                elif (select_dataset == "3. GTZAN Dataset"):
                    dataset = pd.read_csv(f"Dataset/features_30_sec.csv")

            choice2 = st.selectbox(" Dataset or Audio",
                                   ("Dataset Features", "Audio Features"))
            if(choice2 == "Dataset Features"):
                placeholder = st.empty()
                if st.checkbox("Dataset"):
                    placeholder.dataframe(dataset)
                placeholder = st.empty()
                if st.checkbox("Primary Analysis"):
                    dataset_analysis = dataset.label.value_counts().reset_index()
                    placeholder.dataframe(dataset_analysis)
                placeholder = st.empty()
                if st.checkbox("Shape"):
                    dataset_shape = dataset.shape
                    placeholder.write(dataset_shape)
                placeholder = st.empty()
                if st.checkbox("Size"):
                    dataset_size = dataset.size
                    placeholder.write(dataset_size)

            if(choice2 == "Audio Features"):
                # region Song Selection
                col1, col2 = st.beta_columns(2)
                audio_choice = col1.selectbox(
                    " Select Genre ", ("blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"))
                randomfile = random.randint(0, 99)
                if(randomfile >= 0 and randomfile < 10):
                    randomfile2 = str(randomfile)
                    audiofile = audio_choice + ".0000" + randomfile2
                else:
                    randomfile3 = str(randomfile)
                    audiofile = audio_choice + ".000" + randomfile3
                audio_data = 'Data/genres_original/' + audio_choice + "/" + audiofile + ".wav"
                data, sr = librosa.load(audio_data)
                audio_file = open(audio_data, "rb")
                audio_bytes = audio_file.read()
                col2.write("loaded " + audio_choice + " song")
                col2.audio(audio_bytes, format="audio/wav")

                # endregion
                placeholder = st.empty()
                if st.checkbox("Raw Wave"):
                    plt.figure(figsize=(12, 4))
                    librosa.display.waveplot(data, color="#502A75")
                    plt.savefig('Images/rawwave.png')
                    image = Image.open('Images/rawwave.png')
                    st.image(image, caption="Raw Wave")

                placeholder = st.empty()
                if st.checkbox("Spectrogram"):
                    X = librosa.stft(data)
                    Xdb = librosa.amplitude_to_db(abs(X))
                    plt.figure(figsize=(14, 6))
                    librosa.display.specshow(
                        Xdb, sr=sr, x_axis='time', y_axis='log')
                    plt.savefig("Images/colorbar.png")
                    image = Image.open('Images/colorbar.png')
                    st.image(image, caption="Spectrogram I")
                placeholder = st.empty()
                if st.checkbox("Spectral Rolloff"):
                    spectral_rolloff = librosa.feature.spectral_rolloff(
                        data + 0.01, sr=sr)[0]
                    plt.figure(figsize=(12, 4))
                    librosa.display.waveplot(
                        data, sr=sr, alpha=0.4, color="#2B4F72")
                    plt.savefig("Images/spectralrolloff.png")
                    image = Image.open("Images/spectralrolloff.png")
                    st.image(image, caption="Spectral Rolloff")
                placeholder = st.empty()
                if st.checkbox("Zero- Crossing Rate"):
                    n0 = 9000
                    n1 = 9100
                    plt.figure(figsize=(14, 5))
                    plt.plot(data[n0:n1], color="#2B4F72")
                    plt.grid()
                    plt.savefig("Images/zero.png")
                    image = Image.open("Images/zero.png")
                    st.image(image, caption="Zero-Crossing Rate")
                    zero_crossings = librosa.zero_crossings(
                        data[n0:n1], pad=False)
                    st.write("The number of zero-crossings is :",
                             sum(zero_crossings))
                placeholder = st.empty()
                if st.checkbox("Chroma Features"):
                    chromagram = librosa.feature.chroma_stft(data, sr=sr)
                    plt.figure(figsize=(15, 5))
                    librosa.display.specshow(
                        chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
                    plt.savefig("Images/chroma.png")
                    image = Image.open("Images/chroma.png")
                    st.image(image, caption="Chroma Features")

        if (choice == "Create New Dataset[Tutorial]"):
#region codesnippet1
            my_expander = st.beta_expander(label='Code Snippet')
            with my_expander:
                'Copy Code | Change Dataset Name (Line 8) | Select Database Location (Line 17-18) | Run'
                st.markdown(''' # 
                header = 'filename mean(chroma_stft) var(chroma_stft) mean(rms) var(rms) mean(spec_cent) (spec_cent) mean(spec_bw) var(spec_bw) mean(rolloff) var(rolloff) mean(zcr) var(zcr) mean(harmony) var(harmony) tempo'
                for i in range(1, 21):
                    header += f' mean(mfcc{i}) var(mfcc{i})'
                header += ' label'
                header = header.split()

                # Dataset creation function
                file = open('Dataset/data2.csv', 'w', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(header)

                genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
                st.write(genres)
                for g in genres:
                    st.write(f'Genre:{g} started')
                    start = time.clock()
                    for filename in os.listdir(f'Data/genres_original/{g}'):
                        songname = f'Data/genres_original/{g}/{filename}'
                        y, sr = librosa.load(songname, mono=True, sr=None)
                        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                        rms = librosa.feature.rms(y=y)
                        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                        zcr = librosa.feature.zero_crossing_rate(y)
                        mfcc = librosa.feature.mfcc(y=y, sr=sr)
                        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
                        harmony = librosa.feature.tonnetz(y=y, sr=sr)

                        to_append = f'{filename} {np.mean(chroma_stft)} {np.var(chroma_stft)} {np.mean(rms)} {np.var(rms)} {np.mean(spec_cent)} {np.var(spec_cent)} {np.mean(spec_bw)} {np.var(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} {np.var(zcr)} {np.mean(harmony)} {np.var(harmony)} {tempo}'

                        for e in mfcc:
                            to_append += f' {np.mean(e)}'
                            to_append += f' {np.var(e)}'
                        to_append += f' {g}'
                        file = open('Dataset/data2.csv', 'a', newline='')
                        with file:
                            writer = csv.writer(file)
                            writer.writerow(to_append.split())
                    st.write(f'Genre:{g} completed, took {time.clock() - start} seconds') ''')
#endregion
            def newDataset():
                header = 'filename mean(chroma_stft) var(chroma_stft) mean(rms) var(rms) mean(spec_cent) (spec_cent) mean(spec_bw) var(spec_bw) mean(rolloff) var(rolloff) mean(zcr) var(zcr) mean(harmony) var(harmony) tempo'
                for i in range(1, 21):
                    header += f' mean(mfcc{i}) var(mfcc{i})'
                header += ' label'
                header = header.split()

                # Dataset creation function
                file = open('Dataset/datanew.csv', 'w', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(header)

                genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
                st.write(genres)
                for g in genres:
                    st.write(f'Genre:{g} started')
                    start = time.clock()
                    for filename in os.listdir(f'Data/genres_original/{g}'):
                        songname = f'Data/genres_original/{g}/{filename}'
                        y, sr = librosa.load(songname, mono=True, sr=None)
                        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                        rms = librosa.feature.rms(y=y)
                        spec_cent = librosa.feature.spectral_centroid(
                            y=y, sr=sr)
                        spec_bw = librosa.feature.spectral_bandwidth(
                            y=y, sr=sr)
                        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                        zcr = librosa.feature.zero_crossing_rate(y)
                        mfcc = librosa.feature.mfcc(y=y, sr=sr)
                        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
                        harmony = librosa.feature.tonnetz(y=y, sr=sr)

                        to_append = f'{filename} {np.mean(chroma_stft)} {np.var(chroma_stft)} {np.mean(rms)} {np.var(rms)} {np.mean(spec_cent)} {np.var(spec_cent)} {np.mean(spec_bw)} {np.var(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} {np.var(zcr)} {np.mean(harmony)} {np.var(harmony)} {tempo}'

                        for e in mfcc:
                            to_append += f' {np.mean(e)}'
                            to_append += f' {np.var(e)}'
                        to_append += f' {g}'
                        file = open('Dataset/datanew.csv', 'a', newline='')
                        with file:
                            writer = csv.writer(file)
                            writer.writerow(to_append.split())
                    st.write(
                        f'Genre:{g} completed, took {time.clock() - start} seconds')
            if st.button('Create New Dataset'):
                result = newDataset()
            st.write(
                "Warning : Creating new Dataset will take time, to check current Dataset go to Dataset Features")
    if (options == 'Training/Testing'):
        # region PreProcess
        dataset_checkbox = st.selectbox(
            "Select or Upload", ("Select from existing", "Upload Dataset"))
        if (dataset_checkbox == "Upload Dataset"):
            Dataset = st.file_uploader("Upload the Dataset", type=['csv'])
            if Dataset is not None:
                data = pd.read_csv(Dataset)
        elif (dataset_checkbox == "Select from existing"):
            select_dataset = st.selectbox("Select Dataset", (
                "1. Mean Extracted Dataset", "2. Mean and Variance Dataset", "3. GTZAN Dataset"))
            if (select_dataset == "1. Mean Extracted Dataset"):
                data = pd.read_csv(f"Dataset/data.csv")
            elif (select_dataset == "2. Mean and Variance Dataset"):
                data = pd.read_csv(f"Dataset/data2.csv")
            elif (select_dataset == "3. GTZAN Dataset"):
                data = pd.read_csv(f"Dataset/features_30_sec.csv")

#region codesnippet2
        my_expander = st.beta_expander(label='Code Snippet')
        with my_expander:
            'Pre-Processing done on the Dataset before running the model'
            st.markdown(''' # 
        data.head()
        data = data.iloc[0:, 1:]
        y = data['label']
        X = data.loc[:, data.columns != 'label']

        #### NORMALIZE X ####
        cols = X.columns
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(X)
        X = pd.DataFrame(np_scaled, columns=cols)

        #### PCA 2 COMPONENTS ####
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data=principalComponents, columns=[
                                   'principal component 1', 'principal component 2'])

        # concatenate with target label
        finalDf = pd.concat([principalDf, y], axis=1)

        y = data['label']  # genre variable.
        # select all columns but not the labels
        X = data.loc[:, data.columns != 'label']

        #### NORMALIZE X ####

        # Normalize so everything is on the same scale.

        cols = X.columns
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(X)

        # new data frame with the new scaled data.
        X = pd.DataFrame(np_scaled, columns=cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        def model_assess(model, title="Default"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            # print(confusion_matrix(y_test, preds))
            st.write('Accuracy', title, ':', round(
                accuracy_score(y_test, preds), 5), '\')
            filename = title + ".pkl"
            joblib.dump(model, filename)''')

#endregion
        
        data.head()
        data.dropna(inplace=True)
        data = data.iloc[0:, 1:]
        y = data['label']
        X = data.loc[:, data.columns != 'label']

       
        #### PCA 2 COMPONENTS ####
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data=principalComponents, columns=[
                                   'principal component 1', 'principal component 2'])

        # concatenate with target label
        finalDf = pd.concat([principalDf, y], axis=1)

        y = data['label']  # genre variable.
        # select all columns but not the labels
        X = data.loc[:, data.columns != 'label']

        #### NORMALIZE X ####

        # Normalize so everything is on the same scale.


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        # endregion

        def model_assess(model, title="Default"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            # print(confusion_matrix(y_test, preds))
            st.write('Accuracy', title, ':', round(
                accuracy_score(y_test, preds), 5), '\n')
            confusion_matr = confusion_matrix(y_test, preds) #normalize = 'true'
            plt.figure(figsize = (16, 9))
            sns.heatmap(confusion_matr, cmap="Reds", annot=True, 
                xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
                yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);
            plt.savefig("Images/confmatrix.png")
            image = Image.open('Images/confmatrix.png')
            st.image(image, caption="Confusion Matrix for " + title)

        choice = st.selectbox("Select Model for Training/Testing", (
            "Naive Bayes", "Stochastic Gradient Descent", "KNN", "Decision Trees", "Random Forest",
            "Support Vector Machine", "Logistic Regression", "Neural Nets", "XGradientBoost", "XGB(RF)", "CatBoost"))
        if (choice == "Naive Bayes"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitionnb())
            nb = GaussianNB()
            model_assess(nb, "Naive Bayes")
            

        if (choice == "Stochastic Gradient Descent"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitionscd())

            sgd = SGDClassifier(max_iter=5000, random_state=0)
            model_assess(sgd, "Stochastic Gradient Descent")
        if (choice == "KNN"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitionknn())

            knn = KNeighborsClassifier(n_neighbors=19)
            model_assess(knn, "KNN")
        if (choice == "Decision Trees"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitiondt())

            tree = DecisionTreeClassifier()
            model_assess(tree, "Decission trees")
        if (choice == "Random Forest"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitionrf())

            rforest = RandomForestClassifier(
                n_estimators=1000, max_depth=10, random_state=0)
            model_assess(rforest, "Random Forest")
        if (choice == "Support Vector Machine"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitionsvm())

            svm = SVC(decision_function_shape="ovo")
            model_assess(svm, "Support Vector Machine")
        if (choice == "Logistic Regression"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitionlr())

            lg = LogisticRegression(
                random_state=0, solver='lbfgs', multi_class='multinomial')
            model_assess(lg, "Logistic Regression")
        if (choice == "Neural Nets"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitionnn())

            nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(5000, 10), random_state=1)
            model_assess(nn, "Neural Nets")
        if (choice == "XGradientBoost"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitionxgb())

            xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
            model_assess(xgb, "Cross Gradient Booster")
        if (choice == "XGB(RF)"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitionxgbrf())

            xgbrf = XGBRFClassifier(objective='multi:softmax')
            model_assess(xgbrf, "Cross Gradient Booster (Random Forest)")
        if (choice == "CatBoost"):
            placeholder = st.empty()
            if st.checkbox("Definition"):
                placeholder.text(definitioncat())
            cbc = cb.CatBoostClassifier(random_state=seed, verbose=0, eval_metric='Accuracy',
                                        loss_function='MultiClass')
            model_assess(cbc, "CatBoost")

    if(options == 'Genre Recognition'):
        
        songname=None 

        choice = audio_option = st.radio("Select your way",
                                         ("Youtube Url", "Record Audio", "Upload Audio"))

        if (choice == "Youtube Url"):
            
            st.write("using Youtube URL")
            title = st.text_input('Youtube URL', '')
            if title is not '':
                songname = f'./ytdl/songname.wav'
                genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
                genre = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'outtmpl': songname
                    }
                songname = f'./ytdl/songname.wav'
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([title])
                st.audio(songname)
        if (choice == "Record Audio"):
            p = pyaudio.PyAudio()
            FRAMES_PERBUFF = 44100 # number of frames per buffer
            FORMAT = pyaudio.paInt16 # 16 bit int
            CHANNELS = 1 # I guess this is for mono sounds
            FRAME_RATE = 44100 # sample rate

            stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=FRAME_RATE,
                input=True,
                frames_per_buffer=FRAMES_PERBUFF) #buffer

            frames = []

            RECORD_SECONDS = 5
            nchunks = int(RECORD_SECONDS * FRAME_RATE / FRAMES_PERBUFF)
            for i in range(0, nchunks):
                data = stream.read(FRAMES_PERBUFF)
                frames.append(data) # 2 bytes(16 bits) per channel
            st.write("* done recording")
            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open('Audio/recorded_audio.wav', 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(FRAME_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            songname = 'Audio/recorded_audio.wav'
            st.audio(songname)
        if (choice == "Upload Audio"):
            songname = st.file_uploader("Upload the Audio", type=['wav'])
            if songname is not None:
                st.audio(songname)
        def modelTrain():
            data = pd.read_csv(f"Dataset/data.csv")
            data.head()
            data = data.iloc[0:, 1:]
            y = data['label']
            X = data.loc[:, data.columns != 'label']

            

            #### NORMALIZE X ####
            cols = X.columns
            min_max_scaler = preprocessing.MinMaxScaler()
            np_scaled = min_max_scaler.fit_transform(X)
            X = pd.DataFrame(np_scaled, columns=cols)

            #### PCA 2 COMPONENTS ####
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(X)
            principalDf = pd.DataFrame(data=principalComponents, columns=[
                                       'principal component 1', 'principal component 2'])

            # concatenate with target label
            finalDf = pd.concat([principalDf, y], axis=1)

            y = data['label']  # genre variable.
            # select all columns but not the labels
            X = data.loc[:, data.columns != 'label']

            #### NORMALIZE X ####

            # Normalize so everything is on the same scale.

            cols = X.columns
            min_max_scaler = preprocessing.MinMaxScaler()
            np_scaled = min_max_scaler.fit_transform(X)

            # new data frame with the new scaled data.
            X = pd.DataFrame(np_scaled, columns=cols)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42)
            
            st.write("Using CatBoost Classifier")
            cbc = cb.CatBoostClassifier(random_state=seed, verbose=0, eval_metric='Accuracy', loss_function='MultiClass')
            cbc.fit(X_train, y_train)
            preds=cbc.predict(X_test)
            title="CatBoost"
            st.write('Accuracy', title, ':', round(
                accuracy_score(y_test, preds), 5), '\n')
                
            #region ProcessNewAudioandDisplay
            y, sr = librosa.load(songname, mono=True, offset=40.00, duration=30, sr=None)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
            harmony = librosa.feature.tonnetz(y=y, sr=sr)
            row = [np.mean(chroma_stft), np.var(chroma_stft), np.mean(rms), np.var(rms), np.mean(spec_cent),
                   np.var(spec_cent), np.mean(spec_bw), np.var(spec_bw), np.mean(rolloff), np.var(rolloff),
                   np.mean(zcr), np.var(zcr), np.mean(harmony), np.var(harmony), tempo]
            for e in mfcc:
                row.append(np.mean(e))
                row.append(np.var(e))

            X_test = np.asarray(row)
            genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
            genre = [1,2,3,4,5,6,7,8,9,10]
            st.write("CatBoost Recognises it as : ")
            st.write('TYPE:', cbc.predict(X_test)[0])


            plt.figure(figsize=(12, 8))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, y_axis='linear')
            plt.savefig('Images/Output.png')
            image = Image.open('Images/Output.png')
            st.image(image, width=300, caption="Linear-Frequency power spectrogram")
            #endregion
                    
        if songname is not None:           
            if st.button('Train and Recognize'):
                result = modelTrain()
                my_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)
                st.empty()
                songname=None
                st.info('Training Complete')


# endregion
# region Advanced
if(nav == "Advanced"):
    st.title("Advanced")
    st.sidebar.header("Advanced Applications")
    options = st.sidebar.radio(
        "", ("Introduction", "Conversion", "Recommendation", "Transformation", "Creation"))

    if(options == "Introduction"):
        st.header("Introduction")
        st.write("Advanced Applications:")
        st.markdown(''' 
        The webApp is titled, “Music Genre Classification using AI”. Music Genre Classification using AI works on the concept of Music theory, the features of an audio, the core of what makes a music and how its genres are defined, and the classification of music into different genres with the assistance of AI.
        The webApp is divided into multiple components that primarily work on the classification of the genre of music and using that as the foundation, we implement applications that can perform different functions and modifications on and between the music to deepen our technical understanding of audio files. The project would be outsourced as a service that can be used by the general public for purely entertainment and curious reasons, research or by content creators to enhance their workflow or for inspiration.

        The components of the webApp are genre classification, genre recognition, genre recommendation, genre transformation, genre creation/generation and front-end user interface.
        ''')
        st.subheader("Applications :")
        st.write("Conversion :")
        st.write(" --> Select Original and to-Converted Genre")
        st.write(" --> Play Converted Genre File")
        st.write("Recommendation :")
        st.write(" --> Select a genre for a random song")
        st.write(" --> Recommended songs for playing")
        st.write("Transformation :")
        st.write(" --> Random Select |Upload Audio | Youtube Url | Record Audio")
        st.write(" --> Transform the song to your liking from given options.")
        st.write("Creation :")
        st.write(" --> Examples of Music Genre Creation with sample audio")
        st.write(" --> Playable output of sample audio and code snippet.")
    if(options == "Conversion"):

        st.header("Genre Conversion")
        st.write("A basic view, running on streamlit breaks the application. For usage, request through form link given at the end.")
        if st.button("Code to run Locally:"):
            st.markdown('''
  
            CONTENT_FILENAME = audio_data
            STYLE_FILENAME = audio_data2
            first = CONTENT_FILENAME
            second = STYLE_FILENAME
            # Reads wav file and produces spectrum
            # Fourier phases are ignored
            N_FFT = 2048

            def read_audio_spectum(filename):
                x, fs = librosa.load(filename)
                S = librosa.stft(x, N_FFT)
                p = np.angle(S)

                S = np.log1p(np.abs(S[:, :430]))
                return S, fs

            a_content, fs = read_audio_spectum(CONTENT_FILENAME)
            a_style, fs = read_audio_spectum(STYLE_FILENAME)

            N_SAMPLES = a_content.shape[1]
            N_CHANNELS = a_content.shape[0]
            a_style = a_style[:N_CHANNELS, :N_SAMPLES]

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title('Content')
            plt.imshow(a_content[:400, :])
            plt.subplot(1, 2, 2)
            plt.title('Style')
            plt.imshow(a_style[:400, :])
            plt.savefig('content_and_style_spectogram.png')
            plt.close()

            N_FILTERS = 4096

            a_content_tf = np.ascontiguousarray(a_content.T[None, None, :, :])
            a_style_tf = np.ascontiguousarray(a_style.T[None, None, :, :])

            # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
            std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
            kernel = np.random.randn(1, 11, N_CHANNELS, N_FILTERS) * std

            g = tf.Graph()
            with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
                # data shape is "[batch, in_height, in_width, in_channels]",
                x = tf.placeholder(
                    'float32', [1, 1, N_SAMPLES, N_CHANNELS], name="x")

                kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
                conv = tf.nn.conv2d(
                    x,
                    kernel_tf,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                net = tf.nn.relu(conv)

                content_features = net.eval(feed_dict={x: a_content_tf})
                style_features = net.eval(feed_dict={x: a_style_tf})

                features = np.reshape(style_features, (-1, N_FILTERS))
                style_gram = np.matmul(features.T, features) / N_SAMPLES

            from sys import stderr

            ALPHA = 1e-2
            learning_rate = 1e-3
            iterations = 100

            result = None
            with tf.Graph().as_default():

                # Build graph with variable input
                #     x = tf.Variable(np.zeros([1,1,N_SAMPLES,N_CHANNELS], dtype=np.float32), name="x")
                x = tf.Variable(np.random.randn(1, 1, N_SAMPLES, N_CHANNELS).astype(
                    np.float32) * 1e-3, name="x")

                kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
                conv = tf.nn.conv2d(
                    x,
                    kernel_tf,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                net = tf.nn.relu(conv)

                content_loss = ALPHA * 2 * tf.nn.l2_loss(
                    net - content_features)

                style_loss = 0

                _, height, width, number = map(
                    lambda i: i.value, net.get_shape())

                size = height * width * number
                feats = tf.reshape(net, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / N_SAMPLES
                style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

                # Overall loss
                loss = content_loss + style_loss

                opt = tf.contrib.opt.ScipyOptimizerInterface(
                    loss, method='L-BFGS-B', options={'maxiter': 300})

                # Optimization
                with tf.Session() as sess:
                    sess.run(tf.initialize_all_variables())

                    st.write('Started optimization.')
                    opt.minimize(sess)

                    st.write('Final loss:', loss.eval())
                    result = x.eval()

            a = np.zeros_like(a_content)
            a[:N_CHANNELS, :] = np.exp(result[0, 0].T) - 1

            # This code is supposed to do phase reconstruction
            p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
            for i in range(500):
                S = a * np.exp(1j * p)
                x = librosa.istft(S)
                p = np.angle(librosa.stft(x, N_FFT))

            first.split('.')
            second.split('.')
            OUTPUT_FILENAME = 'Audio/converted.wav'
            sf.write(OUTPUT_FILENAME, x, fs, 'PCM_24')

            audio_data = 'Audio/converted.wav'
            data, sr = librosa.load(audio_data)
            audio_file = open(audio_data, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
            plt.figure(figsize=(12, 4))
            librosa.display.waveplot(data, color="#502A75")
            plt.savefig('rawwave1.png')
            image = Image.open('rawwave1.png')
            st.image(image, caption="Waveform III")
            ''')
        col1, col2 = st.beta_columns(2)

        audio_choice = col1.selectbox(" Select Original Genre ", (
            "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"))
        randomfile = random.randint(0, 99)
        if (randomfile >= 0 and randomfile < 10):
            randomfile2 = str(randomfile)
            audiofile = audio_choice + ".0000" + randomfile2
        else:
            randomfile3 = str(randomfile)
            audiofile = audio_choice + ".000" + randomfile3
        audio_data = 'Data/genres_original/' + audio_choice + "/" + audiofile + ".wav"
        data, sr = librosa.load(audio_data)
        audio_file = open(audio_data, "rb")
        audio_bytes = audio_file.read()
        col1.write("loaded " + audio_choice + " song")
        col1.audio(audio_bytes, format="audio/wav")
        plt.figure(figsize=(12, 4))
        librosa.display.waveplot(data, color="#502A75")
        plt.savefig('Images/rawwave01.png')
        image = Image.open('Images/rawwave01.png')
        col1.image(image, caption="Waveform I")
        audio_choice2 = col2.selectbox(" Select Converted Genre ", (
            "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"))
        random2file = random.randint(0, 99)
        if (random2file >= 0 and random2file < 10):
            random2file2 = str(random2file)
            audiofile2 = audio_choice2 + ".0000" + random2file2
        else:
            random2file3 = str(random2file)
            audiofile2 = audio_choice2 + ".000" + random2file3
        audio_data2 = 'Data/genres_original/' + \
            audio_choice2 + "/" + audiofile2 + ".wav"
        data2, sr2 = librosa.load(audio_data2)
        audio_file2 = open(audio_data2, "rb")
        audio_bytes2 = audio_file2.read()
        col2.write("loaded " + audio_choice2 + " song")
        col2.audio(audio_bytes2, format="audio/wav")
        plt.figure(figsize=(12, 4))
        librosa.display.waveplot(data2, color="#502A75")
        plt.savefig('Images/rawwave02.png')
        image = Image.open('Images/rawwave02.png')
        col2.image(image, caption="Waveform II")

        if st.button('Convert'):

            CONTENT_FILENAME = audio_data
            STYLE_FILENAME = audio_data2
            first = CONTENT_FILENAME
            second = STYLE_FILENAME
            # Reads wav file and produces spectrum
            # Fourier phases are ignored
            N_FFT = 2048

            def read_audio_spectum(filename):
                x, fs = librosa.load(filename)
                S = librosa.stft(x, N_FFT)
                p = np.angle(S)

                S = np.log1p(np.abs(S[:, :430]))
                return S, fs

            a_content, fs = read_audio_spectum(CONTENT_FILENAME)
            a_style, fs = read_audio_spectum(STYLE_FILENAME)

            N_SAMPLES = a_content.shape[1]
            N_CHANNELS = a_content.shape[0]
            a_style = a_style[:N_CHANNELS, :N_SAMPLES]

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title('Content')
            plt.imshow(a_content[:400, :])
            plt.subplot(1, 2, 2)
            plt.title('Style')
            plt.imshow(a_style[:400, :])
            plt.savefig('Images/content_and_style_spectogram.png')
            plt.close()

            N_FILTERS = 4096

            a_content_tf = np.ascontiguousarray(a_content.T[None, None, :, :])
            a_style_tf = np.ascontiguousarray(a_style.T[None, None, :, :])

            # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
            std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
            kernel = np.random.randn(1, 11, N_CHANNELS, N_FILTERS) * std

            g = tf.Graph()
            with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
                # data shape is "[batch, in_height, in_width, in_channels]",
                x = tf.placeholder(
                    'float32', [1, 1, N_SAMPLES, N_CHANNELS], name="x")

                kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
                conv = tf.nn.conv2d(
                    x,
                    kernel_tf,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                net = tf.nn.relu(conv)

                content_features = net.eval(feed_dict={x: a_content_tf})
                style_features = net.eval(feed_dict={x: a_style_tf})

                features = np.reshape(style_features, (-1, N_FILTERS))
                style_gram = np.matmul(features.T, features) / N_SAMPLES

            from sys import stderr

            ALPHA = 1e-2
            learning_rate = 1e-3
            iterations = 100

            result = None
            with tf.Graph().as_default():

                # Build graph with variable input
                #     x = tf.Variable(np.zeros([1,1,N_SAMPLES,N_CHANNELS], dtype=np.float32), name="x")
                x = tf.Variable(np.random.randn(1, 1, N_SAMPLES, N_CHANNELS).astype(
                    np.float32) * 1e-3, name="x")

                kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
                conv = tf.nn.conv2d(
                    x,
                    kernel_tf,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                net = tf.nn.relu(conv)

                content_loss = ALPHA * 2 * tf.nn.l2_loss(
                    net - content_features)

                style_loss = 0

                _, height, width, number = map(
                    lambda i: i.value, net.get_shape())

                size = height * width * number
                feats = tf.reshape(net, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / N_SAMPLES
                style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

                # Overall loss
                loss = content_loss + style_loss

                opt = tf.contrib.opt.ScipyOptimizerInterface(
                    loss, method='L-BFGS-B', options={'maxiter': 300})

                # Optimization
                with tf.Session() as sess:
                    sess.run(tf.initialize_all_variables())

                    st.write('Started optimization.')
                    opt.minimize(sess)

                    st.write('Final loss:', loss.eval())
                    result = x.eval()

            a = np.zeros_like(a_content)
            a[:N_CHANNELS, :] = np.exp(result[0, 0].T) - 1

            # This code is supposed to do phase reconstruction
            p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
            for i in range(500):
                S = a * np.exp(1j * p)
                x = librosa.istft(S)
                p = np.angle(librosa.stft(x, N_FFT))

            first.split('.')
            second.split('.')
            OUTPUT_FILENAME = 'Audio/converted.wav'
            sf.write(OUTPUT_FILENAME, x, fs, 'PCM_24')

            audio_data = 'Audio/converted.wav'
            data, sr = librosa.load(audio_data)
            audio_file = open(audio_data, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
            plt.figure(figsize=(12, 4))
            librosa.display.waveplot(data, color="#502A75")
            plt.savefig('Images/rawwave001.png')
            image = Image.open('Images/rawwave001.png')
            st.image(image, caption="Waveform III")

        link = '[Request Genre Conversion!](https://docs.google.com/forms/d/e/1FAIpQLSeGCtdPzKEkGGppVBBflG9m-7rbFAzC1jCVRdnlU1UwtzCe0A/viewform?usp=sf_link)'
        st.markdown(link, unsafe_allow_html=True)

    if(options == "Recommendation"):
        st.header("Genre Recommendation")
        # Libraries
        import IPython.display as ipd
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn import preprocessing
        import pandas as pd
        import numpy as np

        # Read data
        data = pd.read_csv(f'Dataset/features_30_sec.csv', index_col='filename')

        # Extract labels
        labels = data[['label']]

        # Drop labels from original dataframe
        data = data.drop(columns=['label'])

        data.head()

        # Scale the data
        data_scaled = preprocessing.scale(data)
        print('Scaled data type:', type(data_scaled))

        # Cosine similarity
        similarity = cosine_similarity(data_scaled)
        print("Similarity shape:", similarity.shape)

        # Convert into a dataframe and then set the row index and column names as labels
        sim_df_labels = pd.DataFrame(similarity)
        sim_df_names = sim_df_labels.set_index(labels.index)
        sim_df_names.columns = labels.index

        sim_df_names.head()

        def find_similar_songs(name):
            # Find songs most similar to another song
            series = sim_df_names[name].sort_values(ascending=False)

            # Remove cosine similarity == 1 (songs will always have the best match with themselves)
            series = series.drop(name)
            # Display the 5 top matches
            st.write("\n*******\nSimilar songs to ", name)
            st.write(series.head(10))
            series.to_csv('Dataset/temp.csv', index=True)
            simlarityindex=1

        audio_choice = st.selectbox(" Select Genre ", (' ',
            "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"))
        if audio_choice is not ' ':

            randomfile = random.randint(0, 99)
            if (randomfile >= 0 and randomfile < 10):
                randomfile2 = str(randomfile)
                audiofile = audio_choice + ".0000" + randomfile2
            else:
                randomfile3 = str(randomfile)
                audiofile = audio_choice + ".000" + randomfile3
            audio_data = 'Data/genres_original/' + audio_choice + "/" + audiofile + ".wav"
            audio_temp = audiofile + ".wav"
            data, sr = librosa.load(audio_data)
            audio_file = open(audio_data, "rb")
            audio_bytes = audio_file.read()
            st.write("loaded " + audio_choice + " song")
            st.audio(audio_bytes, format="audio/wav")
            similarityindex=None
            if st.button('Find similar songs'):
                find_similar_songs(audio_temp)

                st.write("loaded similiar song")
                df = pd.read_csv('Dataset/temp.csv')
                list=[0,1,2,3,4,5,6,7,8,9]
                for l in list:
                    tempch = df.iloc[l]['filename']
                    temp2 = tempch.split('.')
                    temp=temp2[0]
                    audio_data2 = 'Data/genres_original/' + temp + "/" + tempch
                    data, sr = librosa.load(audio_data2)
                    audio_file2 = open(audio_data2, "rb")
                    audio_bytes2 = audio_file2.read()
                    st.write(tempch)
                    st.audio(audio_bytes2, format="audio/wav")
                

    if(options == "Transformation"):
        st.header("Transformation")
        st.write(" Choose the audio for transformation ")
        
        
        choice = st.radio(
            'Audio', ['Random Selection', 'Upload', 'Youtube', ' Record'])
        songname = None
        if(choice == "Random Selection"):
            audio_choice = st.selectbox(" Select Original Genre ", (
                "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"))
            randomfile = random.randint(0, 99)
            if (randomfile >= 0 and randomfile < 10):
                randomfile2 = str(randomfile)
                audiofile = audio_choice + ".0000" + randomfile2
            else:
                randomfile3 = str(randomfile)
                audiofile = audio_choice + ".000" + randomfile3
            audio_data = 'Data/genres_original/' + audio_choice + "/" + audiofile + ".wav"
            songname = audio_data
            data, sr = librosa.load(audio_data)
            audio_file = open(audio_data, "rb")
            audio_bytes = audio_file.read()
            st.write("loaded " + audio_choice + " song")
            st.audio(audio_bytes, format="audio/wav")
            plt.figure(figsize=(12, 4))
            librosa.display.waveplot(data, color="#502A75")
            plt.savefig('Images/rawwave0.png')
            image = Image.open('Images/rawwave0.png')
            st.image(image, caption="Waveform")
        if(choice == "Upload"):
            audio_data = st.file_uploader('Upload audio', type=["wav"])
            st.audio(audio_data)
            
            data, samplerate = sf.read(audio_data)
            sf.write('Audio/upload_audio.wav', data, samplerate)

            songname='Audio/upload_audio.wav'

            if songname is not None:
                data, sr = librosa.load(songname)
                plt.figure(figsize=(12, 4))
                librosa.display.waveplot(data, color="#502A75")
                plt.savefig('Images/rawwave00.png')
                image = Image.open('Images/rawwave00.png')
                st.image(image, caption="Waveform")
        if(choice == "Youtube"):
            
            st.write("using Youtube URL")
            title = st.text_input('Youtube URL', '')
            if title is not '':
                audio_data = f'Audio/audio_data.wav'
                genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
                genre = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'outtmpl': audio_data
                    }
                audio_data = f'Audio/audio_data.wav'
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([title])
                
                st.audio(audio_data)


                #data, samplerate = sf.read(audio_data)
                #sf.write('youtube_audio.wav', data, samplerate)
                
                songname=audio_data    
           
        if(choice == " Record"):
            if st.button('Record'):
                genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
                genre = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            p = pyaudio.PyAudio()
            FRAMES_PERBUFF = 44100 # number of frames per buffer
            FORMAT = pyaudio.paInt16 # 16 bit int
            CHANNELS = 1 # I guess this is for mono sounds
            FRAME_RATE = 44100 # sample rate

            stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=FRAME_RATE,
                        input=True,
                        frames_per_buffer=FRAMES_PERBUFF) #buffer

            frames = []

            RECORD_SECONDS = 5
            nchunks = int(RECORD_SECONDS * FRAME_RATE / FRAMES_PERBUFF)
            for i in range(0, nchunks):
                data = stream.read(FRAMES_PERBUFF)
                frames.append(data) # 2 bytes(16 bits) per channel
            st.write("* done recording")
            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open('Audio/recorded_audio.wav', 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(FRAME_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            songname = 'Audio/recorded_audio.wav'

            st.audio(songname)
        
        if songname is not None:
            audio=songname
            st.write("Transformation Option")
            AudioSegment.converter = which("ffmpeg")
            audio = AudioSegment.from_file(songname, format="wav")
            choice2 = st.selectbox("Option", ("Pitch Shift", "Loop Song",
                                "Reverse", "Louder/Quieter", "Nightcore", "Librosa Effects"))
            if(choice2 == "Pitch Shift"):
                st.write("Pitch Shift")
                stepsvalue = st.slider("Pitch Shift Scale",min_value=1, max_value=10)
                data, sr = librosa.load(songname)
                data_ps = librosa.effects.pitch_shift(data, sr, n_steps=stepsvalue)
                sf.write('Audio/pitchshift.wav', data_ps, sr, subtype='PCM_24')
                st.audio('Audio/pitchshift.wav')
            if(choice2 == "Loop Song"):
                st.write("Loop Song")
                val = st.slider("Loop songs by", min_value=0, max_value=10)
                loopsong = audio * val
                loopsong.export("Audio/loopsong.wav", format="wav")
                st.audio("Audio/loopsong.wav")
            if(choice2 == "Reverse"):
                st.write("Reverse Song")
                backwards = audio.reverse()
                backwards.export('Audio/reverse.wav', format="wav")
                st.audio('Audio/reverse.wav')
            if(choice2 == "Louder/Quieter"):
                st.write("Louder / Quieter")
                st.write("Scale the loudness and quiteness of the song with slider")
                scale = st.slider("Scale", min_value=-10, max_value=10)
                value = int(scale)
                if(value < 0):
                    st.write("Changing Quiteness by", value, "dB")
                    quieter = audio + value
                    quieter.export("Audio/quieter.wav", format="wav")
                    st.audio("Audio/quieter.wav")
                if(value > 0):
                    st.write("Changing Loudness by", value, "dB")
                    louder = audio + value
                    louder.export('Audio/louder.wav', format="wav")
                    st.audio('Audio/louder.wav')
            if(choice2 == "Nightcore"):
                st.write("Nightcore")

                y, sr = librosa.load(songname)
                y_fast = librosa.effects.time_stretch(y, 1.3)
                y_fast = librosa.effects.pitch_shift(y, sr, n_steps=6)
                sf.write('Audio/nightcore.wav', y_fast, sr, subtype='PCM_24')
                st.audio("Audio/nightcore.wav")

            if(choice2 == "Librosa Effects"):
                st.write("Librosa Effects")
                y, sr = librosa.load(songname)
                y_harmonic = librosa.effects.harmonic(y)
                st.write("Harmonic Values")
                sf.write('Audio/librosa1.wav', y_harmonic, sr, subtype='PCM_24')
                st.audio("Audio/librosa1.wav")

                y_percussive = librosa.effects.percussive(y)
                st.write("Percussive values")
                sf.write('Audio/librosa2.wav', y_percussive, sr, subtype='PCM_24')
                st.audio("Audio/librosa2.wav")
                _, beat_frames = librosa.beat.beat_track(
                    y=y, sr=sr, hop_length=512)
                beat_samples = librosa.frames_to_samples(beat_frames)
                intervals = librosa.util.frame(
                    beat_samples, frame_length=2, hop_length=1).T
                y_out = librosa.effects.remix(y, intervals[::-1])
                st.write("Librosa Remix")
                sf.write('Audio/librosa3.wav', y_out, sr, subtype='PCM_24')
                st.audio("Audio/librosa3.wav")
            if(choice2 == "Vocal Separation"):
                st.write("Vocal Seperation")
                y, sr = librosa.load(songname, duration=30)
                # And compute the spectrogram magnitude and phase
                S_full, phase = librosa.magphase(librosa.stft(y))
                idx = slice(*librosa.time_to_frames([10, 15], sr=sr))
                fig, ax = plt.subplots()
                img = librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                                            y_axis='log', x_axis='time', sr=sr, ax=ax)
                fig.savefig('Images/spectro.png')
                st.image('Images/spectro.png')
                S_filter = librosa.decompose.nn_filter(S_full,
                                                    aggregate=np.median,
                                                    metric='cosine',
                                                    width=int(librosa.time_to_frames(2, sr=sr)))
                margin_i, margin_v = 2, 10
                power = 2

                mask_i = librosa.util.softmask(S_filter,
                                            margin_i * (S_full - S_filter),
                                            power=power)

                mask_v = librosa.util.softmask(S_full - S_filter,
                                            margin_v * S_filter,
                                            power=power)

                # Once we have the masks, simply multiply them with the input spectrum
                # to separate the components

                S_foreground = mask_v * S_full
                S_background = mask_i * S_full
                new_y = librosa.istft(S_foreground * phase)
                sf.write('Audio/vocal.wav', new_y, sr, subtype='PCM_24')
                s1.audio('Audio/vocal.wav')
            if(choice2 == "Extract Background"):
                st.write("Extract Background")
                audio_monoL = audio.split_to_mono()[0]
                audio_monoR = audio.split_to_mono()[1]

                # Invert phase of the Right audio file
                audio_monoR_inv = audio_monoR.invert_phase()

                # Merge two L and R_inv files, this cancels out the centers
                audio_CentersOut = audio_monoL.overlay(audio_monoR_inv)

                # Export merged audio file
                fh = audio_CentersOut.export('Audio/Background.wav', format="wav")
                st.audio('Audio/Background.wav')
    
    
    if(options == "Creation"):
        st.subheader("Creation:")
        st.write("Using Sample Instrument midi file")
        option_instrument=st.selectbox("Keyboard or Keyboard 2", ("Keyboard", "Keyboard 2"))
        if option_instrument=='Keyboard':
            st.write("Sample File-")
            st.audio('Creation/primer.wav')
            st.write("Output using Genre Creation:")
            st.write("First Output:")
            st.audio('Creation/primer_out1.wav')
            st.write("Second Output:")
            st.audio('Creation/primer_out2.wav')
            st.write("Third Output:")
            st.audio('Creation/primer_out3.wav')
        if option_instrument=='Keyboard 2':
            st.write("Sample File-")
            st.audio('Creation/music.wav')
            st.write("Output using Genre Creation:")
            st.write("First Output:")
            st.audio('Creation/music_out1.wav')
            st.write("Second Output:")
            st.audio('Creation/music_out2.wav')
            st.write("Third Output:")
            st.audio('Creation/music_out3.wav')
        
        st.write("Loved what you heard? Request for Genre Creation with sample music")
        link = '[Request Genre Creation!](https://docs.google.com/forms/d/e/1FAIpQLSesvLl8aCNIp9RJwOTKE6DVEl87Pb4tU0HM7mgDP29zS1pI6g/viewform?usp=sf_link)'
        st.markdown(link, unsafe_allow_html=True)

    # endregion
# region Conclusion
if(nav == 'Conclusion'):
    st.title("Conclusion")
    st.header("Thank You!")
    st.header("The webApp will continously progress towards getting better with time.")
    st.header("I hope you liked the applications, it works on desktop and mobile browsers.")
    st.subheader("Also please provide feedback on how it was and if you want something added to it in the future.")
    st.write(' ')
    st.write(' ')
    link = '[Share Feedback!](https://docs.google.com/forms/d/e/1FAIpQLSdSP1CduU8ae9xNUT6jiK1EBJRHkzV2bmSnEEp9fiqP6pFGgQ/viewform?usp=sf_link)'
    st.markdown(link, unsafe_allow_html=True)
    
# endregion
