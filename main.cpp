#include <iostream>
#include "utils.cpp"
#include <vector>

int main(int argc, char **argv) {
    string scene_dir = argv[1];
    // measure the execution time without using opencv
    clock_t start = clock();
    vector<vector<vector<vector<vector<uint8_t>>>>> scene = get_scene_grid(scene_dir);
    vector<string> scene_names = get_scene_names(scene_dir);
    for (int i = 0; i < scene.size(); i++) {
        for (int j = 0; j < scene[i].size(); j++) {
            vector<vector<vector<uint8_t>>> patches = get_frankenpatches(scene,
                                                                         i,
                                                                         j,
                                                                         20,
                                                                         6,
                                                                         1,
                                                                         10);
            string new_name = scene_dir + "/frankenpatches/";

            string filename = scene_names[i * scene[i].size() + j];
            filename = filename.substr(filename.find_last_of("/\\") + 1);
            string::size_type const p(filename.find_last_of('.'));
            filename = filename.substr(0, p) + ".npy";

            new_name += filename;
            cout << new_name << endl;
            save_data(patches, new_name);
        }
    }
    clock_t end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    cout << "Elapsed time: " << elapsed_secs << endl;
    return 0;
}
