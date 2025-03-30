//Setting Page
//Note : This is a placeholder page for later development and not implemented to work with the other pages yet.

import React, { useState } from 'react';
import { View, Text, SafeAreaView, Switch, TextInput, Button, StyleSheet, Alert } from 'react-native';
import ModalSelector from 'react-native-modal-selector';

const SettingsPage = () => {
    const [isDarkMode, setIsDarkMode] = useState(false);
    const [notificationsEnabled, setNotificationsEnabled] = useState(true);
    const [userName, setUserName] = useState('');
    const [language, setLanguage] = useState('English');
    const [privacy, setPrivacy] = useState(true);

    const languages = [
        { key: 1, label: 'English' },
        { key: 2, label: 'Spanish' },
        { key: 3, label: 'French' },
    ];

    const clearData = () => {
        Alert.alert(
            "Clear Data",
            "Are you sure you want to clear all cached data?",
            [
                { text: "Cancel", style: "cancel" },
                { text: "OK", onPress: () => { /* clear cached data logic here */ } },
            ]
        );
    };

    const handleSaveProfile = () => {
        Alert.alert("Profile Saved", `User name: ${userName}`);
    };

    const handleFeedback = () => {
        Alert.alert('Feedback', 'Feedback feature coming soon!');
    };

    const handleLogout = () => {
        Alert.alert(
            "Logout",
            "Are you sure you want to logout?",
            [
                { text: "Cancel", style: "cancel" },
                { text: "Logout", onPress: () => { /* Logic to log out the user */ }},
            ]
        );
    };

    return (
        <View style={styles.container}>
            <SafeAreaView> 
            <Text style={styles.title}>Settings</Text>

            {/* Theme Selection */}
            <View style={styles.settingItem}>
                <Text>Dark Mode</Text>
                <Switch 
                    value={isDarkMode} 
                    onValueChange={setIsDarkMode} 
                />
            </View>
            </SafeAreaView>
            {/* Notification Settings */}
            <View style={styles.settingItem}>
                <Text>Enable Notifications</Text>
                <Switch 
                    value={notificationsEnabled} 
                    onValueChange={setNotificationsEnabled} 
                />
            </View>

            {/* Profile Management */}
            <TextInput 
                style={styles.input} 
                placeholder="Enter your name" 
                value={userName} 
                onChangeText={setUserName} 
            />
            <Button title="Save Profile" onPress={handleSaveProfile} />

            {/* Language Selection */}
            <Text style={styles.label}>Select Language</Text>
            <ModalSelector
                data={languages}
                initValue={language}
                onChange={(option) => setLanguage(option.label)}
            />

            {/* Privacy Settings */}
            <View style={styles.settingItem}>
                <Text>Data Sharing</Text>
                <Switch 
                    value={privacy} 
                    onValueChange={setPrivacy} 
                />
            </View>

            {/* Clear Data */}
            <Button title="Clear Cached Data" onPress={clearData} color="red" />

            {/* Feedback & Support Section */}
            <Button title="Send Feedback" onPress={handleFeedback} />

            {/* Logout Button */}
            <Button title="Logout" onPress={handleLogout} color="red" />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        padding: 20,
        backgroundColor: '#f0f8ff',
    },
    title: {
        fontSize: 28,
        fontWeight: 'bold',
        marginBottom: 20,
    },
    settingItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginVertical: 10,
    },
    input: {
        height: 50,
        borderColor: '#ccc',
        borderWidth: 1,
        marginBottom: 20,
        paddingHorizontal: 15,
        borderRadius: 8,
        backgroundColor: '#ffffff',
    },
    label: {
        marginTop: 20,
        fontSize: 16,
        fontWeight: 'bold',
    }
});

export default SettingsPage;
