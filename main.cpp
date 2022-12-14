#include <iostream>
#include "utils.cpp"
#include <vector>
#include <Eigen/Eigen>

int main(int argc, char** argv) {

    string scene_dir = argv[1];
    // measure the execution time
    clock_t start, end;
    start = clock();
    vector<vector<vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>>> scene = get_scene_grid(scene_dir);
    for (int i = 0; i < scene.size(); i++) {
        for (int j = 0; j < scene[i].size(); j++) {
            vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> patches = get_frankenpatches(scene,
                                                                                                       i,
                                                                                                       j,
                                                                                                       6,
                                                                                                       5,
                                                                                                       1,
                                                                                                       20);
            save_data(patches, "patches.npy");
        }
    }

    end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "Time: " << time << endl;

    return 0;
}
