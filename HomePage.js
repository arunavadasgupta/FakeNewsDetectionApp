

// Homepage


import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, ImageBackground, TouchableOpacity, Image, SafeAreaView } from 'react-native';
import { BlurView } from 'expo-blur'; 
import axios from 'axios';
import { WebView } from 'react-native-webview'; 
import { Modal } from 'react-native'; 

const HomePage = () => {
    const [articles, setArticles] = useState([]); 
    const [videos, setVideos] = useState([]); 
    const [modalVisible, setModalVisible] = useState(false); 
    const [webUrl, setWebUrl] = useState(''); 
    const API_KEY_NEWS = 'API_KY'; //removing API key after getting alert from Github
    const API_KEY_YOUTUBE = 'API_KEY2'; ////removing API key after getting alert from Github

    useEffect(() => {
        const fetchArticles = async () => {
            try {
                const response = await axios.get(`https://newsapi.org/v2/everything?q=fake+news&apiKey=${API_KEY_NEWS}`);
                setArticles(response.data.articles.slice(0, 10)); 
            } catch (error) {
                console.error("Error fetching articles:", error);
            }
        };

        const fetchVideos = async () => {
            try {
                const response = await axios.get(`https://www.googleapis.com/youtube/v3/search?part=snippet&q=fake+news&key=${API_KEY_YOUTUBE}`);
                setVideos(response.data.items); 
            } catch (error) {
                console.error("Error fetching videos:", error);
            }
        };

        fetchArticles();
        fetchVideos();
    }, []);

    const handleLoadMoreArticles = async () => {
        // Logic to fetch more articles to be implemented later
    };

    // to handle scrolling
    const scrollHorizontally = (scrollViewRef, direction) => {
        if (scrollViewRef) {
            scrollViewRef.scrollTo({ x: direction === 'right' ? 200 : -200, animated: true });
        }
    };

    //   handle card press
    const handleCardPress = (url) => {
        setWebUrl(url);
        setModalVisible(true); 
    };

    return (
        <ImageBackground 
            source={require('./assets/jon-tyson-H1flXzFuXgo-unsplash.jpg')}
            style={styles.background}
        >
            <BlurView intensity={5} style={styles.blurContainer}>
            <ScrollView contentContainerStyle={styles.scrollContainer}>
                <SafeAreaView>
                <Text style={styles.title}>Latest In Fake News</Text>
                </SafeAreaView>

                {/* NewsAPI Section */}
                <Text style={styles.sectionTitle}> News API - Latest in Fake News</Text>
                <View style={styles.sectionContainer}>
                    <TouchableOpacity onPress={() => scrollHorizontally(articleScrollRef, 'left')}>
                        <Text style={styles.arrow}>&lt;</Text> 
                    </TouchableOpacity>
                    
                    <ScrollView 
                        horizontal 
                        showsHorizontalScrollIndicator={true} 
                        style={styles.horizontalScroll}
                        ref={ref => articleScrollRef = ref} // Create a reference for scrolling
                    >
                        {articles.map((article, index) => (
                            <TouchableOpacity key={index} onPress={() => handleCardPress(article.url)}>
                                <View style={styles.card}>
                                    <Text style={styles.cardTitle}>{article.title}</Text>
                                    <Text style={styles.cardDescription}>{article.description}</Text>
                                    </View>
                            </TouchableOpacity>
                        ))}
                    </ScrollView>
                    
                    <TouchableOpacity onPress={() => scrollHorizontally(articleScrollRef, 'right')}>
                        <Text style={styles.arrow}>&gt;</Text> 
                    </TouchableOpacity>
                </View>
                <TouchableOpacity onPress={handleLoadMoreArticles} style={styles.moreButton}>
                    <Text style={styles.moreButtonText}>More</Text>
                </TouchableOpacity>

                {/* YouTube Section */}
                <Text style={styles.sectionTitle}>YouTube - Related Videos</Text>
                <View style={styles.sectionContainer}>
                    <TouchableOpacity onPress={() => scrollHorizontally(videoScrollRef, 'left')}>
                        <Text style={styles.arrow}>&lt;</Text> 
                    </TouchableOpacity>

                    <ScrollView 
                        horizontal 
                        showsHorizontalScrollIndicator={true} 
                        style={styles.horizontalScroll}
                        ref={ref => videoScrollRef = ref} // Create a reference for scrolling
                    >
                        {videos.map((video, index) => (
                            <TouchableOpacity key={index} onPress={() => handleCardPress(`https://www.youtube.com/watch?v=${video.id.videoId}`)}>
                                <View style={styles.card}>
                                    <Image 
                                        source={{ uri: video.snippet.thumbnails.high.url }} // Thumbnail URL
                                        style={styles.thumbnail}
                                    />
                                    <Text style={styles.cardTitle}>{video.snippet.title}</Text>
                                </View>
                            </TouchableOpacity>
                        ))}
                    </ScrollView>

                    <TouchableOpacity onPress={() => scrollHorizontally(videoScrollRef, 'right')}>
                        <Text style={styles.arrow}>&gt;</Text>
        
                    </TouchableOpacity>
                </View>

                {/* More Coming Soon Section */}
                <Text style={styles.sectionTitle}>More Coming Soon!</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                    {[...Array(5)].map((_, index) => (
                        <View key={index} style={styles.card}>
                            <Text style={styles.cardTitle}>Placeholder Card {index + 1}</Text>
                        </View>
                    ))}
                </ScrollView>
                
                {/* Modal for WebView */}
                <Modal 
                    visible={modalVisible} 
                    animationType="slide" 
                    onRequestClose={() => setModalVisible(false)} // Close the modal
                >
                    <WebView source={{ uri: webUrl }} style={{ flex: 1 }} />
                    <TouchableOpacity style={styles.closeButton} onPress={() => setModalVisible(false)}>
                        <Text style={styles.closeButtonText}>Close</Text>
                    </TouchableOpacity>
                </Modal>

            </ScrollView>
            
            </BlurView>
        </ImageBackground>
    );
};


const styles = StyleSheet.create({
    background: {
        flex: 1,
        justifyContent: 'center',
    },
    scrollContainer: {
        flexGrow: 0,
        alignItems: 'center',
        padding: 20,
    },
    blurContainer: {
        flex: 1, 
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.3)',
    },
    title: {
        fontSize: 28,
        fontWeight: 'bold',
        marginBottom: 20,
        color: '#ffffff', 
        textAlign: 'center',
    },
    sectionTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        marginVertical: 10,
        color: '#ffffff', 
        backgroundColor: 'rgba(0, 0, 0, 0.6)', 
        padding: 10,
        borderRadius: 10,
    },
    sectionContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 10,
    },
    arrow: {
        fontSize: 24,
        color: '#ffffff', 
        paddingHorizontal: 10,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        borderRadius: 15,
    },
    card: {
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        borderRadius: 10,
        padding: 15,
        shadowColor: '#000',
        shadowOffset: {
            width: 0,
            height: 2,
        },
        shadowOpacity: 0.3,
        shadowRadius: 6,
        elevation: 4, 
        marginHorizontal: 10,
        width: 200, 
    },
    cardTitle: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
    },
    cardDescription: {
        fontSize: 12,
        color: '#555',
        marginTop: 5,
    },
    moreButton: {
        marginVertical: 10,
        padding: 10,
        borderRadius: 5,
        backgroundColor: 'rgba(255, 255, 255, 0.7)',
        alignItems: 'center',
    },
    moreButtonText: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
    },
    thumbnail: {
        width: 180,  
        height: 100,
        borderRadius: 10,
    },
    horizontalScroll: {
        marginBottom: 10,
    },
    closeButton: {
        backgroundColor: 'rgba(0, 0, 0, 0.7)', 
        padding: 10,
        borderRadius: 5,
        position: 'absolute',
        top: 40,
        right: 20,
    },
    closeButtonText: {
        color: '#ffffff',
        fontSize: 16,
        fontWeight: 'bold',
    },
});

export default HomePage;


