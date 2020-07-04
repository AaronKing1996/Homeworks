import train
import tools
import configparser


if __name__ == "__main__":
    # 参数配置
    config = configparser.ConfigParser()
    config.read('config.ini')

    train.train(config)
    pass
