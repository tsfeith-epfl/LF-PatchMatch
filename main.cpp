#include <iostream>
#include "utils.cpp"
#include <vector>
#include <Eigen/Eigen>

int main(int argc, char** argv) {

    string scene_dir = argv[1];
    // measure the execution time
    vector<vector<vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>>> scene = get_scene_grid(scene_dir);
    vector<string> scene_names = get_scene_names(scene_dir);
    for (int i = 0; i < scene.size(); i++) {
        for (int j = 0; j < scene[i].size(); j++) {
            vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> patches = get_frankenpatches(scene,
                                                                                                       i,
                                                                                                       j,
                                                                                                       6,
                                                                                                       5,
                                                                                                       1,
                                                                                                       20);
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
    return 0;
}
