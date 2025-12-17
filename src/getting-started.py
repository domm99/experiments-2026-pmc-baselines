from ProFed.partitioner import Environment, Region, download_dataset, split_train_validation, partition_to_subregions


if __name__ == "__main__":

    number_of_subregions = 5
    number_of_devices_per_subregion = 10

    mapping_devices_area = {
        i: list(range(number_of_devices_per_subregion*i, number_of_devices_per_subregion*i+number_of_devices_per_subregion))
        for i in range(number_of_subregions)
    }

    train_data, test_data = download_dataset('EMNIST')
    train_data, validation_data = split_train_validation(train_data, 0.8)

    environment = partition_to_subregions(train_data, validation_data, 'EMNIST', 'Hard', number_of_subregions, seed=42)

    mapping = {}

    for region_id, devices in mapping_devices_area.items():
        mapping_devices_data = environment.from_subregion_to_devices(region_id, len(devices))
        for device_index, data in mapping_devices_data.items():
            device_id = devices[device_index]
            mapping[device_id] = data

    print(mapping)