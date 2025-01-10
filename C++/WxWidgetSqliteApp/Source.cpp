#include <iostream>
#include <sqlite3.h>

int main() {
    sqlite3* db;
    char* errMessage = 0;

    // Open a database connection
    int exit = sqlite3_open("example.db", &db);
    if (exit) {
        std::cerr << "Error open DB: " << sqlite3_errmsg(db) << std::endl;
        return exit;
    } else {
        std::cout << "Opened Database Successfully!" << std::endl;
    }

    // Create a SQL table
    const char* sqlCreateTable = "CREATE TABLE IF NOT EXISTS Users ("
                                  "ID INTEGER PRIMARY KEY AUTOINCREMENT, "
                                  "Name TEXT NOT NULL, "
                                  "Age INTEGER NOT NULL);";

    exit = sqlite3_exec(db, sqlCreateTable, 0, 0, &errMessage);
    if (exit != SQLITE_OK) {
        std::cerr << "Error Create Table: " << errMessage << std::endl;
        sqlite3_free(errMessage);
    } else {
        std::cout << "Table created successfully!" << std::endl;
    }

    // Insert data into the table
    const char* sqlInsert = "INSERT INTO Users (Name, Age) VALUES ('Alice', 30);"
                            "INSERT INTO Users (Name, Age) VALUES ('Bob', 25);";

    exit = sqlite3_exec(db, sqlInsert, 0, 0, &errMessage);
    if (exit != SQLITE_OK) {
        std::cerr << "Error Insert Data: " << errMessage << std::endl;
        sqlite3_free(errMessage);
    } else {
        std::cout << "Data inserted successfully!" << std::endl;
    }

    // Query the data
    const char* sqlSelect = "SELECT * FROM Users;";
    sqlite3_stmt* stmt;

    exit = sqlite3_prepare_v2(db, sqlSelect, -1, &stmt, 0);
    if (exit != SQLITE_OK) {
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
    } else {
        std::cout << "Query results:" << std::endl;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(stmt, 0);
            const unsigned char* name = sqlite3_column_text(stmt, 1);
            int age = sqlite3_column_int(stmt, 2);
            std::cout << "ID: " << id << ", Name: " << name << ", Age: " << age << std::endl;
        }
    }

    // Finalize the statement
    sqlite3_finalize(stmt);

    // Close the database connection
    sqlite3_close(db);
    return 0;
}