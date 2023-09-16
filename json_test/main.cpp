#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // a JSON text
    char text[] = R"(
    {
    "Object": [
        {
            "idx": 0,
            "id": 2,
            "distance":  10.0
        },
        {
            "idx": 1,
            "id": 10,
            "distance":  10.0
        },
        {
            "idx": 2,
            "id": 10,
            "distance":  10.0
        },
        {
            "idx": 3,
            "id": 10,
            "distance":  10.0
        }]
    }
    )";

    // parse and serialize JSON
    json js_test = json::parse(text);

    auto json_array_test = js_test["Object"];

    for(auto js : json_array_test) {
        std::cout << std::setw(4) << js << "\n\n";
        std::cout << "distance: " << js["distance"] << "\n";

    }
}
