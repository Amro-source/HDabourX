#include <iostream>
#include "pugixml.hpp"

int main() {
    // Create a document object
    pugi::xml_document doc;

    // Load the XML file
    pugi::xml_parse_result result = doc.load_file("example.xml");

    // Check if the file was loaded successfully
    if (!result) {
        std::cerr << "Error loading XML file: " << result.description() << std::endl;
        return 1;
    }

    // Access the root node
    pugi::xml_node root = doc.child("root");

    // Iterate through child nodes
    for (pugi::xml_node child = root.first_child(); child; child = child.next_sibling()) {
        std::cout << "Child name: " << child.attribute("name").value() << std::endl;
        std::cout << "Child value: " << child.child_value() << std::endl;
    }

    // Modify an element
    pugi::xml_node child1 = root.child("child");
    child1.text().set("NewValue1");

    // Save the modified XML to a new file
    doc.save_file("modified_example.xml");

    return 0;
}