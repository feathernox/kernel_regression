if __name__ == '__main__':
    delta_K = 3
    d = [10, 20, 50, 100, 200, 500, 1000, 2000]
    conf_str = [f'{{d: {d_val}, n:{int(delta_K * d_val)}}}' for d_val in d]
    print(','.join(conf_str))
