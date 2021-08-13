import image_caption
# import image_folder

# 如果添加了新的数据集代码，请自行扩充此字典(导入新增数据集, 并添加在字典里)
# 此字典将被用于建立数据集索引，可根据名称直接加载数据集
dataset_dict = {
    "image_caption": image_caption,
    # "image_folder": image_folder,
}


# 根据输入的"字符串"在"dataset_dict"中获取对应的"数据集文件"进行加载
def get_data_by_name(name: str, **kwargs):
    if name.islower() and not name.startswith("_"):
        # 根据名称加载对应的数据集
        data_file = dataset_dict[name]
        # 每个数据集中都必须包含[train_dataset, train_dataset, train_dataset]三个函数
        # 用于此处加载训练集、测试集、验证集。
        train_dataset = data_file.train_dataset(**kwargs)
        val_dataset = data_file.val_dataset(**kwargs)
        test_dataset = data_file.test_dataset(**kwargs)

    else:
        print("[ERROR] Data name you selected is not support, but can be registered.")
        print("[WARNING] Custom dataset loading file should be add in 'core/dataset/data/*', "
              "and import in file 'core/dataset/data/__init__.py'.")
        raise NameError

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    print("[INFO] Searching Dataset...")
    train, val, test = get_data_by_name("image_caption")
    print(train, val, test)
    print("[INFO] Done.")
