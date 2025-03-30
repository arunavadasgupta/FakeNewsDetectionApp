//Image Classification Page

import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Image, ActivityIndicator, Alert, ImageBackground } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { BlurView } from 'expo-blur';

const ImagePage = () => {
    const [image, setImage] = useState(null);
    const [realProbability, setRealProbability] = useState(0);
    const [fakeProbability, setFakeProbability] = useState(0);
    const [classification, setClassification] = useState('');
    const [loading, setLoading] = useState(false);

    const pickImage = async () => {
        const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (permission.granted === false) {
            alert("Sorry, we need camera roll permissions to make this work!");
            return;
        }

        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 1,
        });

        if (!result.canceled) {
            setImage(result.assets[0].uri);  // Set the selected image
            classifyImage(result.assets[0].uri);
        }
    };

    const classifyImage = async (imageUri) => {
        setLoading(true);
        setClassification('');
        setRealProbability(0);
        setFakeProbability(0);

        const formData = new FormData();
        formData.append('image', {
            uri: imageUri,
            name: 'image.jpg',
            type: 'image/jpeg',
        });

        try {
            const response = await axios.post('http://127.0.0.1:5001/classify_image', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setRealProbability(response.data.probability_real);
            setFakeProbability(response.data.probability_fake);
            setClassification(response.data.classification);
        } catch (error) {
            console.error("Error classifying image:", error);
            Alert.alert('Error', 'Failed to classify the image. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const clearImage = () => {
        setImage(null);
        setClassification('');
        setRealProbability(0);
        setFakeProbability(0);
    };

    return (
        <ImageBackground 
            source={require('./assets/jon-tyson-H1flXzFuXgo-unsplash.jpg')}
            style={styles.background} 
        >
        <BlurView intensity={12} style={styles.blurView}>

            <View style={styles.container}>
                    <Text style={styles.title}>Image Classification</Text>

                    {image && (
                        <View style={styles.imagePreviewContainer}>
                            <Image source={{ uri: image }} style={styles.imagePreview} />
                            <TouchableOpacity onPress={clearImage} style={styles.clearButton}>
                                <Text style={styles.clearButtonText}>Clear Image</Text>
                            </TouchableOpacity>
                        </View>
                    )}

                    {loading && <ActivityIndicator size="large" color="#0000ff" />}

                    {(realProbability > 0 || fakeProbability > 0) && (
                        <View style={styles.resultSection}>
                            <Text style={styles.resultText}>Probabilities:</Text>
                            <Text style={styles.resultPercent}>Real: {(realProbability * 100).toFixed(2)}%</Text>
                            <Text style={styles.resultPercent}>Fake: {(fakeProbability * 100).toFixed(2)}%</Text>
                            {classification && (
                                <Text style={[
                                    styles.classificationText,
                                    { color: classification === 'Fake' ? 'darkred' : 'darkgreen' }
                                ]}>
                                    This image appears to be {classification.toUpperCase()}!
                                </Text>
                            )}
                        </View>
                    )}

                {/* "Pick an Image" Button at the Bottom */}
                <TouchableOpacity onPress={pickImage} style={styles.cameraButton}>
                    <Text style={styles.cameraText}>Pick an Image</Text>
                </TouchableOpacity>
            </View>
            </BlurView>
        </ImageBackground>
    );
};

const styles = StyleSheet.create({
    background: {
        flex: 1,
        justifyContent: 'center', 
    },
    container: {
        flex: 0,
        justifyContent: 'flex-start',
        alignItems: 'center',
        padding: 20,
    },
    blurView: {
        flex: 1, 
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.3)',
    },
    title: {
        fontSize: 28,
        fontWeight: 'bold',
        marginBottom: 20,
        color: '#333',
        textAlign: 'center',
    },
    imagePreviewContainer: {
        alignItems: 'center',
        marginTop: 20,
    },
    imagePreview: {
        width: 200,
        height: 200,
        borderRadius: 10,
    },
    clearButton: {
        backgroundColor: '#FF5733', 
        padding: 10,
        borderRadius: 10,
        marginTop: 10,
    },
    clearButtonText: {
        color: '#ffffff',
        fontSize: 16,
    },
    resultSection: {
        marginTop: 20,
        alignItems: 'center',
    },
    resultText: {
        fontSize: 25,
        fontWeight: 'bold',
    },
    resultPercent:{
        fontSize: 20,
        fontWeight: 'bold',
    },
    classificationText: {
        marginTop: 10,
        fontSize: 22, 
        fontWeight: 'bold',
        color: '#ffffff', 
        padding: 5, 
        borderRadius: 8, 
    },
    cameraButton: {
        marginTop: 30,
        backgroundColor: '#007BFF',
        padding: 15,
        borderRadius: 10,
        alignItems: 'center',
    },
    cameraText: {
        color: '#ffffff',
        fontSize: 16,
    },
});

export default ImagePage;

