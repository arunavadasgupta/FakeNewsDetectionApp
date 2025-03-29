
import React, { useState } from 'react';
import {
    View,
    Text,
    TextInput,
    Button,
    ScrollView,
    StyleSheet,
    ActivityIndicator,
    Alert,
    Modal,
    Image,
    Switch,
    ImageBackground
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import ModalSelector from 'react-native-modal-selector';
import axios from 'axios';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing'; // Optional for sharing files

const TextPage = () => {
    const [text, setText] = useState('');
    const [modelType, setModelType] = useState('Transformer');
    const [nlpModel, setnlpModel] = useState('Logistic Regression');
    const [selectedTransformerModel, setSelectedTransformerModel] = useState('DistilBERT'); 
    const [result, setResult] = useState({});
    const [loading, setLoading] = useState(false);
    const [modelFile, setModelFile] = useState(null);
    const [modalVisible, setModalVisible] = useState(false); 
    const [trainingModalVisible, setTrainingModalVisible] = useState(false); 
    const [vectorizerFile, setVectorizerFile] = useState(null);

    // 

    const classifyText = async () => {
        setLoading(true);
        setResult({});
        
        try {
            const formData = new FormData();
            
            // model type based on the selected option
            let modelTypeToSend;
            if (modelType === 'Transformer') {
                modelTypeToSend = selectedTransformerModel.toLowerCase(); 
            } else {
                modelTypeToSend = 'nlp'; 
            }
    
            // Check if using the uploaded model and vectorizer
            if (modelFile && vectorizerFile && text.trim()) {
                formData.append('model', {
                    uri: modelFile.uri,
                    name: modelFile.name,
                    type: 'application/octet-stream',
                });
                formData.append('vectorizer', {
                    uri: vectorizerFile.uri,
                    name: vectorizerFile.name,
                    type: 'application/octet-stream',
                });
                
                formData.append('text', text);
                formData.append('model_type', modelTypeToSend); 
            } else if (text.trim()) {
                formData.append('text', text);
                formData.append('model_type', modelTypeToSend); 
            }
    
            const response = await axios.post('http://127.0.0.1:5001/classify', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
    
            const { classification, probability } = response.data;
            setResult({ classification, probability });
            setModalVisible(true);
            
        } catch (error) {
            console.error(error.response ? error.response.data : error.message);
            Alert.alert('Error', 'Failed to classify the text.');
        } finally {
            setLoading(false);
        }
    };
    
    

    const useDefaultModel = async () => {
        setLoading(true);
        try {
            const response = await axios.post('http://127.0.0.1:5001/train', {
                model_type: nlpModel, 
            });
            setResult(response.data); 
            setTrainingModalVisible(true); 
        } catch (error) {
            console.error(error);
            Alert.alert('Error', 'Failed to train the default model.');
        } finally {
            setLoading(false);
        }
    };

    const uploadFile = async () => {
        if (!modelFile) {
            Alert.alert('File Required', 'Please upload a model file.');
            return;
        }

        setLoading(true);
        const formData = new FormData();
        formData.append('file', {
            uri: modelFile.uri,
            name: modelFile.name,
            type: 'text/csv',
        });

        try {
            const response = await axios.post('http://127.0.0.1:5001/train', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setResult(response.data);
            setTrainingModalVisible(true); 
        } catch (error) {
            console.error(error);
            Alert.alert('Error', 'Failed to upload and train the model.');
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async () => {
        const result = await DocumentPicker.getDocumentAsync({ type: 'text/csv' });
        if (result.type === 'success') {
            setModelFile(result);
            uploadFile();
        } else {
            Alert.alert('File Upload Cancelled', 'No file was selected.');
        }
    };

    const saveFile = async (fileName, fileUri) => {
        const fileUriToSave = `http://127.0.0.1:5001/${fileUri}`;
    
        try {
            const downloadsDirectory = FileSystem.documentDirectory + '/Files/Downloads/';
            await FileSystem.makeDirectoryAsync(downloadsDirectory, { intermediates: true });
    
            const fileInfo = await FileSystem.downloadAsync(
                fileUriToSave,
                downloadsDirectory + fileName
            );
    
            Alert.alert('File Saved', `The file has been saved to your device: ${fileInfo.uri}`);
        } catch (error) {
            console.error(error);
            Alert.alert('Error', 'Failed to save file.');
        }
    };

    const handleFileSave = () => {
        if (result.model_file) {
            saveFile('trained_model.pkl', result.model_file);
        }
        if (result.vectorizer_file) {
            saveFile('tfidf_vectorizer.pkl', result.vectorizer_file);
        }
    };

    const clearText = () => {
        setText('');
    };

    const resetSettings = () => {
        setText('');
        setModelType('Transformer');
        setnlpModel('Logistic Regression');
        setSelectedTransformerModel('DistilBERT'); 
        setResult({});
        setModelFile(null);
        setVectorizerFile(null);
    };

    const handleModelFileUpload = async () => {
        const result = await DocumentPicker.getDocumentAsync({ type: 'application/octet-stream' });
        if (result.type === 'success') {
            setModelFile(result);
        } else {
            Alert.alert('File Upload Cancelled', 'No model file was selected.');
        }
    };
    
    const handleVectorizerFileUpload = async () => {
        const result = await DocumentPicker.getDocumentAsync({ type: 'application/octet-stream' });
        if (result.type === 'success') {
            setVectorizerFile(result);
        } else {
            Alert.alert('File Upload Cancelled', 'No vectorizer file was selected.');
        }
    };

    return (

        <ImageBackground 
        source={require('./assets/jon-tyson-H1flXzFuXgo-unsplash.jpg')}
        style={styles.background}
    >
        <SafeAreaView style={styles.container}>
            <Text style={styles.title}>Fake News Classification</Text>
            <View style={styles.classificationModeContainer}>
                <Text style={styles.label}>
                Current Classification Mode: {modelType}
                </Text>
                {modelType === 'Transformer' && (
                <Text style={styles.selectedModel}>
                Model: {selectedTransformerModel}
                </Text>
            )}
            </View>

            <TextInput
                style={styles.textInput}
                placeholder="Enter your text here..."
                value={text}
                onChangeText={setText}
                multiline
                numberOfLines={6}
            />
            <Button style={styles.buttonStyle} title="Classify Text" onPress={classifyText} disabled={loading || text.length === 0} />
            {text.length > 0 && <Button style={styles.buttonStyle} title="Clear Text" onPress={clearText} />}
            {loading && <ActivityIndicator size="large" color="#0000ff" />}
            
            {result.classification && (
                <Text style={{
                    color: result.classification === 'Fake' ? 'darkred' : 'darkgreen',
                    marginTop: 10,
                    fontSize: 18,
                    textAlign: 'center',
                    fontWeight:'bold'
                }}>
                    This content is classified as: {result.classification}
                    {'\n'}Probability Score: {result.probability !== undefined ? result.probability.toFixed(2) : 'N/A'}
                </Text>
            )}

            <Text style={styles.label}>Select Model Type</Text>
            <Switch
                value={modelType === 'Transformer'}
                onValueChange={value => {
                    setModelType(value ? 'Transformer' : 'NLP');
                    setnlpModel(value ? '' : 'Logistic Regression');
                }}
            />
            <Text style={styles}>{modelType} Selected</Text>

            {modelType === 'NLP' && (
                <>
                    <Text style={styles.label}>Select NLP Model</Text>
                    <ModalSelector
                        data={[
                            { key: 1, label: 'Logistic Regression' },
                            { key: 2, label: 'Random Forest' },
                            { key: 3, label: 'Support Vector Machine' },
                        ]}
                        initValue={nlpModel}
                        onChange={(option) => {
                            console.log(`Selected model: ${option.label}`); 
                            setnlpModel(option.label); 
                            }}
                            style={styles.selectorContainer}  
                            selectTextStyle={styles.selectorText}
                            initValueTextStyle={styles.selectorText}
                                            
                    />
                    <Button style={styles.buttonStyle} title="Upload Training Data" onPress={handleFileUpload} disabled={loading} />
                    <Button style={styles.buttonStyle} title="Train Using Default Data" onPress={useDefaultModel} />
                </>
            )}

            {modelType === 'Transformer' && (
                <>
                    <Text style={styles.label}>Select Transformer Model</Text>
                    <ModalSelector
                        data={[
                            { key: 1, label: 'DistilBERT' },
                            { key: 2, label: 'BERT' },
                            { key: 3, label: 'RoBERTa' },
                            { key: 4, label: 'XLNet' },
                            { key: 5, label: 'ALBERT' },
                        ]}
                        initValue={selectedTransformerModel}
                        onChange={(option) => setSelectedTransformerModel(option.label)}
                        style={styles.selectorContainer}  // Use enhanced style
                        selectTextStyle={styles.selectorText}
                        initValueTextStyle={styles.selectorText}
                    />
                </>
            )}

            <Modal
                animationType="slide"
                transparent={true}
                visible={trainingModalVisible}
                onRequestClose={() => setTrainingModalVisible(false)}
            >
                <View style={styles.modalContainer}>
                    <View style={styles.modalContent}>
                        <Text style={styles.modalTitle}>Results</Text>
                        
                        {result.message && (
                            <View style={styles.resultTextContainer}>
                                <Text style={styles.resultText}>Message: {result.message}</Text>
                                <Text>Accuracy: {result.accuracy ? result.accuracy.toFixed(2) : 'N/A'}</Text>
                                <Text>Precision: {result.precision ? result.precision.toFixed(2) : 'N/A'}</Text>
                                <Text>Recall: {result.recall ? result.recall.toFixed(2) : 'N/A'}</Text>
                                <Text>F1 Score: {result.f1_score ? result.f1_score.toFixed(2) : 'N/A'}</Text>
                                <Text>Confusion Matrix: {JSON.stringify(result.cm)}</Text>
                            </View>
                        )}

                        <Image
                            source={{ uri: 'http://127.0.0.1:5001/assets/confusion_matrix.png' }}
                            style={styles.image}
                        />
                        <Image
                            source={{ uri: 'http://127.0.0.1:5001/assets/roc_curve.png' }}
                            style={styles.image}
                        />
                        
                        <Button title="Save Model" onPress={handleFileSave} />
                        <Button title="Close" onPress={() => setTrainingModalVisible(false)} />
                    </View>
                </View>
            </Modal>

            <Button style={styles.buttonStyle} title="Upload Model File" onPress={handleModelFileUpload} disabled={loading} />
            <Button title="Upload Vectorizer File" onPress={handleVectorizerFileUpload} disabled={loading} />

            {modelFile && vectorizerFile ? (
                <Button title="Classify Text Using Uploaded Model" onPress={classifyText} />
            ) : (
                <Button title="Classify Text Using Uploaded Model" onPress={() => Alert.alert("Please upload both model and vectorizer files first.")} disabled />
            )}

            {loading && <ActivityIndicator size="large" color="#0000ff" />}

            <Button title="Reset Settings" onPress={resetSettings} />
        </SafeAreaView>
        </ImageBackground>
    );
};

const styles = StyleSheet.create({
    background: {
        flex: 1,
        justifyContent: 'center',
        },
    container: {
        flex: 1,
        justifyContent: 'flex-start',
        padding: 20,
        backgroundColor: 'rgba(240, 248, 255, 0.7)', 
        borderRadius: 15,
        shadowColor: '#000',
        shadowOffset: {
            width: 0,
            height: 5,
        },
        shadowOpacity: 0.25,
        shadowRadius: 6.27,
        elevation: 10, 
    },
    title: {
        fontSize: 28,
        fontWeight: 'bold',
        textAlign: 'center',
        marginVertical: 20,
    },
    textInput: {
        height: 100,
        borderColor: '#ccc',
        borderWidth: 1,
        marginBottom: 20,
        paddingHorizontal: 15,
        borderRadius: 8,
        backgroundColor: '#ffffff',
    },
    modalContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.5)', 
    },
    modalContent: {
        width: '80%',
        backgroundColor: 'white',
        padding: 20,
        borderRadius: 10,
        alignItems: 'center',
    },
    modalTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 20,
    },
    resultTextContainer: {
        marginBottom: 15,
    },
    resultText: {
        fontSize: 16,
        textAlign: 'left',
        marginBottom: 5,
    },
    image: {
        width: '100%',
        height: 200,
        resizeMode: 'contain',
        marginVertical: 10,
    },
    label:{
        fontWeight: 'bold',

    },
    selectedModel:{
        fontWeight: 'bold',
        marginTop:10,
        marginBlock:10
        
    }, 
    buttonStyle:{
        fontWeight: 'bold',
        fontSize: 30,
    },
    selectorContainer: {
        borderWidth: 2, 
        borderColor: 'transparent', 
        backgroundColor: 'rgba(255, 255, 255, 0.5)', 
        borderRadius: 10,
        padding: 5, 
        marginTop: 10, 
    },
    selectorText: {
        fontSize: 18,
        color: '#333', 
        fontWeight: 'bold',
        textAlign: 'center', 
    },
    
});

export default TextPage;

