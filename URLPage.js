//URL CLassification Page

import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, ImageBackground,ActivityIndicator } from 'react-native';
import { BlurView } from 'expo-blur';
import axios from 'axios';
import { SafeAreaView } from 'react-native-safe-area-context';

const URLPage = () => {
    const [url, setUrl] = useState('');
    const [result, setResult] = useState('');
    const [loading, setLoading] = useState(false);

    const classifyURL = async () => {
        setLoading(true);
        setResult('');


        try {
            if (url.trim() === '') {
                Alert.alert('Input Required', 'Please enter a URL.');
                return;
            }
            const regex = /^(http|https):\/\/.+/;
            if (!regex.test(url)) {
                Alert.alert('Invalid URL', 'Please enter a valid URL starting with http:// or https://');
                setLoading(false);
                return;
            }     


            const response = await axios.post('http://127.0.0.1:5001/classify', 
            new URLSearchParams({ url }).toString(), {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
            });            
            const { classification } = response.data; 
            console.log("Classifying URL:", url);
            setResult(classification === 1 ? 'The content is classified as Fake News.' : 'The content is classified as Real News.');
        } catch (error) {
            console.error("Error details:", error.response ? error.response.data : error.message);
            setResult('Error fetching or classifying content.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <ImageBackground 
            source={require('./assets/jon-tyson-H1flXzFuXgo-unsplash.jpg')}
            style={styles.background}
        >
            <View style={styles.container}>
                <SafeAreaView>
                    <Text style={styles.title}>URL Classifier</Text>
                    <TextInput
                        style={styles.input}
                        placeholder="Enter news article URL"
                        value={url}
                        onChangeText={setUrl}
                    />
                    <Button style={styles.buttonStyle} title={loading ? 'Classifying...' : 'Classify URL'} onPress={classifyURL} disabled={loading} />
                    {loading && <ActivityIndicator size="large" color="#0000ff" />}
                </SafeAreaView>
                <ScrollView style={styles.resultContainer}>
                    {result && <Text style={styles.result}>{result}</Text>}
                </ScrollView>    
            </View>
        </ImageBackground>
    );
};

const styles = StyleSheet.create({
    background: {
        flex: 1,
        justifyContent: 'center',
        },
    blurContainer: {
        padding: 20,
        borderRadius: 15,
        alignItems: 'center',
        justifyContent: 'flex-start',
        marginVertical: 10,
        backgroundColor: 'transparent',
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
        elevation: 10, // For Android shadow
    },
    title: {
        fontSize: 28,
        fontWeight: 'bold',
        marginBottom: 20,
        color: '#333',
        textAlign: 'center',
    },
    input: {
        height: 50,
        borderColor: '#ccc',
        borderWidth: 1,
        marginBottom: 20,
        paddingHorizontal: 15,
        borderRadius: 8,
        backgroundColor: 'rgba(255, 255, 255, 0.9)', 
    },
    resultContainer: {
        marginTop: 20,
        padding: 15,
        borderRadius: 8,
        backgroundColor: 'rgba(255, 255, 255, 0.8)', 
        shadowColor: '#000',
        shadowOffset: {
            width: 0,
            height: 2,
        },
        shadowOpacity: 0.2,
        shadowRadius: 2.62,
        elevation: 4,
    },
    result: {
        fontSize: 22,
        fontWeight:'bold',
        textAlign: 'center',
        color: '#333',
    },
    buttonStyle:{
        fontWeight:'bold'
    }


});

export default URLPage;
