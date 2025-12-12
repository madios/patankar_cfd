//
// Created by Peter Berg Ammundsen on 12/12/2025.
//

#ifndef TODO_H
#define TODO_H
#include <fstream>
#include <iosfwd>
#include <vector>

namespace Todo
{
    inline void saveAsCSV(const std::vector<double>& vec, const std::string& filename) {
        std::ofstream file(filename);
        for (size_t i = 0; i < vec.size(); ++i)
        {
            file << vec[i] << "\n";
        }
        file.close();
        //saveAsCSV(field,"/Users/ri03jm/Library/Mobile Documents/com~apple~CloudDocs/DropboxFolder/Peter/skole/PhD/MatlabScript/output.csv");
    }
}



#endif //TODO_H
