#include <iostream>
#include "utils.cpp"
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

void compute_save_patches(const std::string& scene_dir,
                          const vector<vector<vector<vector<vector<uint8_t>>>>>& scene,
                          const vector<string>& scene_names,
                          int i,
                          int j,
                          int patch_size,
                          int num_patches,
                          int stride,
                          int roi) {
    /* Compute and save patches for a given scene and for view i, j
       This function is separated from main to allow parallelization */
    // main part of the function, compute the frankenpatches
    vector<vector<vector<uint8_t>>> patches = get_frankenpatches(scene,
                                                                         i,
                                                                         j,
                                                                         patch_size,
                                                                         num_patches,
                                                                         stride,
                                                                         roi);
    string new_name = scene_dir + "/frankenpatches/";

    // get the filename of the original scene, and change the extension to .npy
    string filename = scene_names[i * scene[i].size() + j];
    filename = filename.substr(filename.find_last_of("/\\") + 1);
    string::size_type const p(filename.find_last_of('.'));
    filename = filename.substr(0, p) + ".npy";

    // save the patches
    new_name += filename;
    save_data(patches, new_name);
}

int main(int argc, char **argv) {
    string scene_dir = argv[1];
    int grid_size_0 = stoi(argv[2]);
    int grid_size_1 = stoi(argv[3]);

    int patch_size = stoi(argv[4]);
    int num_patches = stoi(argv[5]);
    int stride = stoi(argv[6]);
    int roi = stoi(argv[7]);

    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

    // get the views as uint8_t and the names of each view
    vector<vector<vector<vector<vector<uint8_t>>>>> scene = get_scene_grid(scene_dir, grid_size_0, grid_size_1);
    vector<string> scene_names = get_scene_names(scene_dir, grid_size_0, grid_size_1);

    #pragma omp parallel for default(none) shared(scene_dir, scene, scene_names, patch_size, num_patches, stride, roi)
    for (int i = 0; i < scene.size(); i++) {
        for (int j = 0; j < scene[i].size(); j++) {
            compute_save_patches(scene_dir, scene, scene_names, i, j, patch_size, num_patches, stride, roi);
        }
    }

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    // get the time in milliseconds
    auto duration = chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();
    cout << "Time taken: " << duration << " milliseconds" << endl;
    return 0;
}
