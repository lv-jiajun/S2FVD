












def main():
    pre = 'a'
    count = 3000000
    filename ='E:/model/JiajunLv/code1/test.txt'
    f_new = open('file_part.txt', 'w', encoding='utf-8')
    n = 1
    with open(filename, "r+", encoding="utf8") as file:
        flag = 0
        for line in file:
            if (count % 3000 == 0):
                f_new.close()
                f_new = open('file_part' + str(count/3000) + '.cpp', 'w', encoding='utf-8')
            stripped = line.strip()
            if not stripped:          #判断是否是空，如果是则continue
                continue
            elif "-" * 26 in line:
                flag = 1
                f_new.write('\n')
                f_new.write('\n')
                count = count + 1
                continue
            elif flag == 1:
                kv = line.strip().split('(')
                line = line.replace(kv[0], str(pre)+str(count)+kv[0])
                f_new.write(line)
                flag = 0
            elif str("CWE-119") in stripped:
                continue
            elif str("CWE-120") in stripped:
                continue
            elif str("CWE-469") in stripped:
                continue
            elif str("CWE-476") in stripped:
                continue
            else:
                f_new.write(line)


if __name__ == "__main__":
    main()