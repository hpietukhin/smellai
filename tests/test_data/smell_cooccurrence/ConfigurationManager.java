package com.example.smells;

import java.util.Map;
import java.util.HashMap;
import java.util.Properties;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Demonstrates: Multiple interconnected smells showing complex dependencies
 * - Long Method
 * - Long Parameter List
 * - Duplicated Code
 * - Print Statements
 * - Switch Statement
 * - Complex Method
 *
 * Shows how fixing one smell can cascade to fix others (positive dependencies)
 */
public class ConfigurationManager {

    private Map<String, String> configuration = new HashMap<>();
    private Properties properties = new Properties();

    // LONG METHOD + DUPLICATED CODE + PRINT STATEMENTS + SWITCH STATEMENT
    public boolean loadConfiguration(String environment, String configPath, String region,
                                    boolean useCache, boolean validateSchema, boolean encryptSecrets,
                                    String backupPath, int retryCount) { // LONG PARAMETER LIST (8 params)
        System.out.println("Loading configuration for environment: " + environment); // Print Statement

        // Duplicated validation block #1
        if (environment == null || environment.trim().isEmpty()) {
            System.out.println("Error: Environment cannot be null");
            return false;
        }
        if (configPath == null || configPath.trim().isEmpty()) {
            System.out.println("Error: Config path cannot be null");
            return false;
        }
        if (region == null || region.trim().isEmpty()) {
            System.out.println("Error: Region cannot be null");
            return false;
        }

        // Large SWITCH STATEMENT
        String configFile;
        switch (environment) {
            case "development":
                configFile = configPath + "/dev-config.properties";
                System.out.println("Loading development configuration");
                if (useCache) {
                    System.out.println("Cache enabled for development");
                }
                break;
            case "testing":
                configFile = configPath + "/test-config.properties";
                System.out.println("Loading testing configuration");
                if (useCache) {
                    System.out.println("Cache enabled for testing");
                }
                break;
            case "staging":
                configFile = configPath + "/staging-config.properties";
                System.out.println("Loading staging configuration");
                if (validateSchema) {
                    System.out.println("Schema validation enabled for staging");
                }
                break;
            case "production":
                configFile = configPath + "/prod-config.properties";
                System.out.println("Loading production configuration");
                if (validateSchema) {
                    System.out.println("Schema validation enabled for production");
                }
                if (encryptSecrets) {
                    System.out.println("Secret encryption enabled for production");
                }
                break;
            default:
                System.out.println("Unknown environment: " + environment);
                return false;
        }

        // Duplicated error handling
        try {
            FileInputStream fis = new FileInputStream(configFile);
            properties.load(fis);
            fis.close();
            System.out.println("Configuration loaded successfully from: " + configFile);
        } catch (IOException e) {
            System.out.println("Error loading configuration: " + e.getMessage());
            if (retryCount > 0) {
                System.out.println("Retrying... attempts left: " + retryCount);
                return loadConfiguration(environment, configPath, region, useCache,
                                       validateSchema, encryptSecrets, backupPath, retryCount - 1);
            }
            return false;
        }

        // Another switch statement for region-specific settings
        switch (region) {
            case "us-east":
                configuration.put("server", "us-east-server.example.com");
                configuration.put("cdn", "us-east-cdn.example.com");
                System.out.println("US East region configured");
                break;
            case "us-west":
                configuration.put("server", "us-west-server.example.com");
                configuration.put("cdn", "us-west-cdn.example.com");
                System.out.println("US West region configured");
                break;
            case "eu-central":
                configuration.put("server", "eu-central-server.example.com");
                configuration.put("cdn", "eu-central-cdn.example.com");
                System.out.println("EU Central region configured");
                break;
            case "asia-pacific":
                configuration.put("server", "asia-pacific-server.example.com");
                configuration.put("cdn", "asia-pacific-cdn.example.com");
                System.out.println("Asia Pacific region configured");
                break;
            default:
                System.out.println("Unknown region: " + region);
                return false;
        }

        System.out.println("Configuration loading complete");
        return true;
    }

    // LONG METHOD + LONG PARAMETER LIST + DUPLICATED CODE
    public boolean updateConfiguration(String environment, String configPath, String region,
                                      String key, String value, boolean validateValue,
                                      boolean createBackup, String backupPath) { // 8 params
        System.out.println("Updating configuration key: " + key);

        // Duplicated validation block #2 (same as in loadConfiguration)
        if (environment == null || environment.trim().isEmpty()) {
            System.out.println("Error: Environment cannot be null");
            return false;
        }
        if (configPath == null || configPath.trim().isEmpty()) {
            System.out.println("Error: Config path cannot be null");
            return false;
        }
        if (region == null || region.trim().isEmpty()) {
            System.out.println("Error: Region cannot be null");
            return false;
        }

        if (key == null || key.trim().isEmpty()) {
            System.out.println("Error: Key cannot be null");
            return false;
        }

        if (validateValue) {
            // Duplicated validation logic
            if (value == null || value.trim().isEmpty()) {
                System.out.println("Error: Value cannot be null");
                return false;
            }
        }

        if (createBackup) {
            // Duplicated backup logic
            try {
                System.out.println("Creating backup at: " + backupPath);
                // Backup logic here
            } catch (Exception e) {
                System.out.println("Error creating backup: " + e.getMessage());
                return false;
            }
        }

        configuration.put(key, value);
        System.out.println("Configuration updated successfully");
        return true;
    }

    // LONG PARAMETER LIST + DUPLICATED CODE
    public String getConfiguration(String environment, String configPath, String region,
                                  String key, String defaultValue, boolean useCache,
                                  boolean logAccess) { // 7 params
        System.out.println("Getting configuration key: " + key);

        // Duplicated validation block #3
        if (environment == null || environment.trim().isEmpty()) {
            System.out.println("Error: Environment cannot be null");
            return defaultValue;
        }
        if (configPath == null || configPath.trim().isEmpty()) {
            System.out.println("Error: Config path cannot be null");
            return defaultValue;
        }
        if (region == null || region.trim().isEmpty()) {
            System.out.println("Error: Region cannot be null");
            return defaultValue;
        }

        if (logAccess) {
            System.out.println("Access logged for key: " + key + " in environment: " + environment);
        }

        String value = configuration.get(key);
        if (value == null) {
            System.out.println("Key not found, returning default value");
            return defaultValue;
        }

        return value;
    }

    // DUPLICATED CODE - Similar error handling in multiple methods
    public boolean deleteConfiguration(String environment, String configPath, String region,
                                      String key, boolean createBackup, String backupPath) {
        System.out.println("Deleting configuration key: " + key);

        // Duplicated validation again
        if (environment == null || environment.trim().isEmpty()) {
            System.out.println("Error: Environment cannot be null");
            return false;
        }
        if (configPath == null || configPath.trim().isEmpty()) {
            System.out.println("Error: Config path cannot be null");
            return false;
        }
        if (region == null || region.trim().isEmpty()) {
            System.out.println("Error: Region cannot be null");
            return false;
        }

        // Duplicated backup logic
        if (createBackup) {
            try {
                System.out.println("Creating backup at: " + backupPath);
                // Backup logic here
            } catch (Exception e) {
                System.out.println("Error creating backup: " + e.getMessage());
                return false;
            }
        }

        configuration.remove(key);
        System.out.println("Configuration deleted successfully");
        return true;
    }
}
