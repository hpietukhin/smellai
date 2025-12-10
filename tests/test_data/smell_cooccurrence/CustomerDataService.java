package com.example.smells;

import java.util.Date;

/**
 * Demonstrates: Long Parameter List, Feature Envy, Data Class (negative dependency)
 *
 * Negative Dependency:
 * - The methods have Long Parameter List (8+ parameters)
 * - To fix this, we would use "Introduce Parameter Object" refactoring
 * - This creates CustomerData class below - which becomes a DATA CLASS smell
 * - Data Class has only getters/setters with no behavior
 *
 * This shows the trade-off: fixing Long Parameter List creates Data Class
 */
public class CustomerDataService {

    private DatabaseConnection connection;
    private EmailService emailService;

    // LONG PARAMETER LIST smell (8 parameters)
    public void createCustomerAccount(String firstName, String lastName, String email,
                                     String phone, String address, String city,
                                     String state, String zipCode) {
        // Feature Envy - using emailService more than own data
        if (emailService.isValidEmail(email)) {
            emailService.sendWelcomeEmail(email, firstName);
        }

        // Feature Envy - using connection more than own data
        connection.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            firstName, lastName, email, phone, address, city, state, zipCode
        );
    }

    // LONG PARAMETER LIST smell (9 parameters)
    public void updateCustomerProfile(int customerId, String firstName, String lastName,
                                     String email, String phone, String address,
                                     String city, String state, String zipCode) {
        // Feature Envy
        String oldEmail = connection.query("SELECT email FROM customers WHERE id = ?", customerId);

        if (!email.equals(oldEmail)) {
            // Feature Envy - more interaction with emailService
            emailService.sendEmailChangeNotification(oldEmail, email);
            emailService.sendEmailVerification(email, firstName);
        }

        // Feature Envy
        connection.execute(
            "UPDATE customers SET first_name=?, last_name=?, email=?, phone=?, " +
            "address=?, city=?, state=?, zip=? WHERE id=?",
            firstName, lastName, email, phone, address, city, state, zipCode, customerId
        );
    }

    // LONG PARAMETER LIST smell (10 parameters)
    public boolean validateCustomerData(String firstName, String lastName, String email,
                                       String phone, String address, String city,
                                       String state, String zipCode, Date birthDate,
                                       String ssn) {
        if (firstName == null || firstName.trim().isEmpty()) return false;
        if (lastName == null || lastName.trim().isEmpty()) return false;
        if (email == null || !emailService.isValidEmail(email)) return false;
        if (phone == null || phone.length() < 10) return false;
        if (address == null || address.trim().isEmpty()) return false;
        if (city == null || city.trim().isEmpty()) return false;
        if (state == null || state.length() != 2) return false;
        if (zipCode == null || zipCode.length() != 5) return false;
        if (birthDate == null || birthDate.after(new Date())) return false;
        if (ssn == null || ssn.length() != 9) return false;

        return true;
    }

    // LONG PARAMETER LIST smell (8 parameters)
    public String formatCustomerAddress(String firstName, String lastName, String address,
                                       String city, String state, String zipCode,
                                       String country, boolean includeCountry) {
        StringBuilder formatted = new StringBuilder();
        formatted.append(firstName).append(" ").append(lastName).append("\n");
        formatted.append(address).append("\n");
        formatted.append(city).append(", ").append(state).append(" ").append(zipCode);

        if (includeCountry) {
            formatted.append("\n").append(country);
        }

        return formatted.toString();
    }
}

/**
 * DATA CLASS smell - Negative dependency from fixing Long Parameter List
 *
 * This class would be created when refactoring the Long Parameter List above
 * using "Introduce Parameter Object" pattern.
 *
 * Problem: This class contains only data (fields + getters/setters) with NO BEHAVIOR.
 * It's a pure data holder, which is considered a code smell.
 *
 * This demonstrates the NEGATIVE DEPENDENCY:
 * - Solving Long Parameter List → Creates Data Class
 */
class CustomerData {
    private String firstName;
    private String lastName;
    private String email;
    private String phone;
    private String address;
    private String city;
    private String state;
    private String zipCode;
    private Date birthDate;
    private String ssn;

    // Only getters and setters - NO BEHAVIOR
    public String getFirstName() { return firstName; }
    public void setFirstName(String firstName) { this.firstName = firstName; }

    public String getLastName() { return lastName; }
    public void setLastName(String lastName) { this.lastName = lastName; }

    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }

    public String getPhone() { return phone; }
    public void setPhone(String phone) { this.phone = phone; }

    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }

    public String getCity() { return city; }
    public void setCity(String city) { this.city = city; }

    public String getState() { return state; }
    public void setState(String state) { this.state = state; }

    public String getZipCode() { return zipCode; }
    public void setZipCode(String zipCode) { this.zipCode = zipCode; }

    public Date getBirthDate() { return birthDate; }
    public void setBirthDate(Date birthDate) { this.birthDate = birthDate; }

    public String getSsn() { return ssn; }
    public void setSsn(String ssn) { this.ssn = ssn; }
}

// Stub classes for compilation
class DatabaseConnection {
    public void execute(String sql, Object... params) {}
    public String query(String sql, Object... params) { return ""; }
}

class EmailService {
    public boolean isValidEmail(String email) { return true; }
    public void sendWelcomeEmail(String email, String name) {}
    public void sendEmailChangeNotification(String oldEmail, String newEmail) {}
    public void sendEmailVerification(String email, String name) {}
}
