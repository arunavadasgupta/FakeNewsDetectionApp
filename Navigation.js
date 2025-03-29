import React from 'react';
import { TouchableOpacity } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import HomePage from './HomePage';
import URLPage from './URLPage';        // Import for URL classification page
import ImagePage from './ImagePage';    // Import for Image classification page
import ScanInput from './ScanInput';
import TextPage from './TextPage';      // Import for Text classification page
import SettingsPage from './SettingsPage'; // Import for Settings
import { MaterialIcons } from 'react-native-vector-icons'; // Import icon library


const Tab = createBottomTabNavigator();

const AppNavigator = () => {
    return (
        <NavigationContainer>
            <Tab.Navigator
                screenOptions={{
                    headerShown: false,
                    tabBarStyle: { backgroundColor: '#ffffff', borderTopWidth: 0 }
                }}
            >
                <Tab.Screen 
                    name="Home" 
                    component={HomePage} 
                    options={{
                        tabBarIcon: ({ color, size }) => (
                            <MaterialIcons name="home" color={color} size={size} />
                        ),
                    }} 
                />
                <Tab.Screen 
                    name="URL" 
                    component={URLPage} 
                    options={{
                        tabBarIcon: ({ color, size }) => (
                            <MaterialIcons name="link" color={color} size={size} />
                        ),
                    }} 
                />
                {/* <Tab.Screen 
                    name="Scan" 
                    component={ScanInput} 
                    options={{
                        tabBarIcon: ({ color }) => (
                            <MaterialIcons name="camera-alt" color={color} size={60} /> // Larger icon for scanning
                        ),
                        tabBarButton: (props) => (
                            <TouchableOpacity {...props} style={{ marginBottom: 20 }} /> // Adjust styling
                        ),
                    }} 
                /> */}
                <Tab.Screen 
                    name="Text" 
                    component={TextPage} 
                    options={{
                        tabBarIcon: ({ color, size }) => (
                            <MaterialIcons name="text-fields" color={color} size={size} />
                        ),
                    }} 
                />
                <Tab.Screen 
                    name="Image" 
                    component={ImagePage} // For the image classification
                    options={{
                        tabBarIcon: ({ color, size }) => (
                            <MaterialIcons name="camera-alt" color={color} size={size} />
                        ),
                    }} 
                />
                <Tab.Screen 
                    name="Settings" 
                    component={SettingsPage} 
                    options={{
                        tabBarIcon: ({ color, size }) => (
                            <MaterialIcons name="settings" color={color} size={size} />
                        ),
                    }} 
                />
            </Tab.Navigator>
        </NavigationContainer>
    );
};

export default AppNavigator;
