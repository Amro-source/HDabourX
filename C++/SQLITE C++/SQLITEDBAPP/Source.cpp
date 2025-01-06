#include <iostream>
#include <sqlite3.h> // SQLite header file

// Function to execute an SQL statement and display results (callback)
static int callback(void* NotUsed, int argc, char** argv, char** azColName) {
    for (int i = 0; i < argc; i++) {
        std::cout << azColName[i] << ": " << (argv[i] ? argv[i] : "NULL") << std::endl;
    }
    std::cout << std::endl;
    return 0;
}

int main() {
    sqlite3* db;
    char* errMsg = nullptr;

    // Open database (or create it if it doesn't exist)
    int rc = sqlite3_open("example.db", &db);
    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return rc;
    }
    else {
        std::cout << "Opened database successfully!" << std::endl;
    }

    // SQL statement to create a table
    const char* createTableSQL = R"(
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL
        );
    )";

    rc = sqlite3_exec(db, createTableSQL, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }
    else {
        std::cout << "Table created successfully!" << std::endl;
    }

    // Insert data into the table
    const char* insertSQL = R"(
        INSERT INTO users (name, age) VALUES ('Alice', 25), ('Bob', 30), ('Charlie', 22);
    )";

    rc = sqlite3_exec(db, insertSQL, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }
    else {
        std::cout << "Records inserted successfully!" << std::endl;
    }

    // Query data from the table
    const char* selectSQL = "SELECT * FROM users;";
    std::cout << "Querying data from the users table..." << std::endl;
    rc = sqlite3_exec(db, selectSQL, callback, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }

    // Close the database
    sqlite3_close(db);
    std::cout << "Database connection closed." << std::endl;

    return 0;
}
