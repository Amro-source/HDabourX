#include <iostream>
#include <sqlite3.h>
#include <string>

// Function prototypes
void displayMenu();
void createTable(sqlite3* db);
void addItem(sqlite3* db);
void viewItems(sqlite3* db);
void updateItem(sqlite3* db);
void deleteItem(sqlite3* db);

// Callback function for displaying query results
static int callback(void* NotUsed, int argc, char** argv, char** azColName) {
    for (int i = 0; i < argc; i++) {
        std::cout << azColName[i] << ": " << (argv[i] ? argv[i] : "NULL") << "\t";
    }
    std::cout << std::endl;
    return 0;
}

int main() {
    sqlite3* db;
    int choice;

    // Open or create database
    if (sqlite3_open("inventory.db", &db)) {
        std::cerr << "Error opening database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    std::cout << "Welcome to the Inventory Management System!" << std::endl;

    // Main application loop
    do {
        displayMenu();
        std::cin >> choice;

        switch (choice) {
        case 1:
            createTable(db);
            break;
        case 2:
            addItem(db);
            break;
        case 3:
            viewItems(db);
            break;
        case 4:
            updateItem(db);
            break;
        case 5:
            deleteItem(db);
            break;
        case 6:
            std::cout << "Exiting application. Goodbye!" << std::endl;
            break;
        default:
            std::cout << "Invalid choice. Please try again." << std::endl;
        }
    } while (choice != 6);

    // Close the database
    sqlite3_close(db);
    return 0;
}

void displayMenu() {
    std::cout << "\n==== Inventory Management System ====" << std::endl;
    std::cout << "1. Create Inventory Table" << std::endl;
    std::cout << "2. Add Item to Inventory" << std::endl;
    std::cout << "3. View All Items" << std::endl;
    std::cout << "4. Update Item Details" << std::endl;
    std::cout << "5. Delete Item from Inventory" << std::endl;
    std::cout << "6. Exit" << std::endl;
    std::cout << "Enter your choice: ";
}

void createTable(sqlite3* db) {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            category TEXT
        );
    )";

    char* errMsg = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }
    else {
        std::cout << "Inventory table created successfully!" << std::endl;
    }
}

void addItem(sqlite3* db) {
    std::string name, category;
    int quantity;
    double price;

    std::cout << "Enter item name: ";
    std::cin.ignore();
    std::getline(std::cin, name);
    std::cout << "Enter quantity: ";
    std::cin >> quantity;
    std::cout << "Enter price: ";
    std::cin >> price;
    std::cout << "Enter category: ";
    std::cin.ignore();
    std::getline(std::cin, category);

    std::string sql = "INSERT INTO inventory (name, quantity, price, category) VALUES ('" +
        name + "', " + std::to_string(quantity) + ", " +
        std::to_string(price) + ", '" + category + "');";
    char* errMsg = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }
    else {
        std::cout << "Item added successfully!" << std::endl;
    }
}

void viewItems(sqlite3* db) {
    const char* sql = "SELECT * FROM inventory;";
    char* errMsg = nullptr;

    std::cout << "\nInventory Items:" << std::endl;
    if (sqlite3_exec(db, sql, callback, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }
}

void updateItem(sqlite3* db) {
    int id, quantity;
    double price;
    std::string name, category;

    std::cout << "Enter the ID of the item to update: ";
    std::cin >> id;
    std::cout << "Enter new name: ";
    std::cin.ignore();
    std::getline(std::cin, name);
    std::cout << "Enter new quantity: ";
    std::cin >> quantity;
    std::cout << "Enter new price: ";
    std::cin >> price;
    std::cout << "Enter new category: ";
    std::cin.ignore();
    std::getline(std::cin, category);

    std::string sql = "UPDATE inventory SET name = '" + name + "', quantity = " +
        std::to_string(quantity) + ", price = " + std::to_string(price) +
        ", category = '" + category + "' WHERE id = " + std::to_string(id) + ";";
    char* errMsg = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }
    else {
        std::cout << "Item updated successfully!" << std::endl;
    }
}

void deleteItem(sqlite3* db) {
    int id;
    std::cout << "Enter the ID of the item to delete: ";
    std::cin >> id;

    std::string sql = "DELETE FROM inventory WHERE id = " + std::to_string(id) + ";";
    char* errMsg = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }
    else {
        std::cout << "Item deleted successfully!" << std::endl;
    }
}
