from pycparser import c_parser, parse_file

parser = c_parser.CParser()

# code = "void CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket_02_bad ( ) { int data ; data = - 1 ; if ( 1 ) { { struct wsaData ; int wsaDataInit = 0 ; int recvResult ; struct sockaddr_in service ; int connectSocket = INVALID_int ; char inputBuffer [ ( 3 * sizeof ( data ) + 2 ) ] ; do { if ( WSAStartup ( MAKEWORD ( 2 , 2 ) , & wsaData ) != NO_ERROR ) { break ; } wsaDataInit = 1 ; connectSocket = socket ( AF_INET , SOCK_STREAM , IPPROTO_TCP ) ; if ( connectSocket == INVALID_int ) { break ; } memset ( & service , 0 , sizeof ( service ) ) ; service . sin_family = AF_INET ; service . sin_addr . s_addr = inet_addr ( \"127.0.0.1\" ) ; service . sin_port = htons ( 27015 ) ; if ( connect ( connectSocket , ( struct sockaddr * ) & service , sizeof ( service ) ) == - 1 ) { break ; } recvResult = recv ( connectSocket , inputBuffer , ( 3 * sizeof ( data ) + 2 ) - 1 , 0 ) ; if ( recvResult == - 1 || recvResult == 0 ) { break ; } inputBuffer [ recvResult ] = '\0' ; data = atoi ( inputBuffer ) ; } while ( 0 ) ; if ( connectSocket != INVALID_int ) { close ( connectSocket ) ; } if ( wsaDataInit ) { WSACleanup ( ) ; } } } if ( 1 ) { { int i ; int buffer [ 10 ] = { 0 } ; if ( data >= 0 ) { buffer [ data ] = 1 ; for ( i = 0 ; i < 10 ; i ++ ) { printIntLine ( buffer [ i ] ) ; } } else { printLine ( \"ERROR: Array index is negative.\" ) ; } } } } "

code = r"""
typedef struct _charVoid
{
    char charFirst[16];
    void * voidSecond;
    void * voidThird;
} charVoid;
void CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01_bad()
{
    {
        charVoid structCharVoid;
        structCharVoid.voidSecond = (void *)SRC_STR;
        printLine((char *)structCharVoid.voidSecond);
        memcpy(structCharVoid.charFirst, SRC_STR, sizeof(structCharVoid));
        structCharVoid.charFirst[(sizeof(structCharVoid.charFirst)/sizeof(char))-1] = '\0';
        printLine((char *)structCharVoid.charFirst);
        printLine((char *)structCharVoid.voidSecond);
    }
}
"""
# code = "void CWE121_Stack_Based_Buffer_Overflow__CWE129_fscanf_67b_badSink ( struct myStruct ) { int data = myStruct . structFirst ; { int i ; int buffer [ 10 ] = { 0 } ; if ( data >= 0 ) { buffer [ data ] = 1 ; for ( i = 0 ; i < 10 ; i ++ ) { printIntLine ( buffer [ i ] ) ; } } else { printLine ( \"ERROR: Array index is negative.\" ) ; } } } "

# code = "void CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket_02_bad ( ) { int data ; data = - 1 ; if ( 1 ) { { struct wsaData ; int wsaDataInit = 0 ; int recvResult ; struct sockaddr_in service ; int connectSocket = INVALID_int ; char inputBuffer [ ( 3 * sizeof ( data ) + 2 ) ] ; do { if ( WSAStartup ( MAKEWORD ( 2 , 2 ) , & wsaData ) != NO_ERROR ) { break ; } wsaDataInit = 1 ; connectSocket = socket ( AF_INET , SOCK_STREAM , IPPROTO_TCP ) ; if ( connectSocket == INVALID_int ) { break ; } memset ( & service , 0 , sizeof ( service ) ) ; service . sin_family = AF_INET ; service . sin_addr . s_addr = inet_addr ( \"127.0.0.1\" ) ; service . sin_port = htons ( 27015 ) ; if ( connect ( connectSocket , ( struct sockaddr * ) & service , sizeof ( service ) ) == - 1 ) { break ; } recvResult = recv ( connectSocket , inputBuffer , ( 3 * sizeof ( data ) + 2 ) - 1 , 0 ) ; if ( recvResult == - 1 || recvResult == 0 ) { break ; } inputBuffer [ recvResult ] = '\0' ; data = atoi ( inputBuffer ) ; } while ( 0 ) ; if ( connectSocket != INVALID_int ) { close ( connectSocket ) ; } if ( wsaDataInit ) { WSACleanup ( ) ; } } } if ( 1 ) { { int i ; int buffer [ 10 ] = { 0 } ; if ( data >= 0 ) { buffer [ data ] = 1 ; for ( i = 0 ; i < 10 ; i ++ ) { printIntLine ( buffer [ i ] ) ; } } else { printLine ( \"ERROR: Array index is negative.\" ) ; } } } } "

code = "typedef struct _charVoid{    char charFirst[16];    void * voidSecond;    void * voidThird;} charVoid;static const int STATIC_CONST_TRUE = 1; static const int STATIC_CONST_FALSE = 0; void CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_04_bad(){    if(STATIC_CONST_TRUE)    {        {            charVoid structCharVoid;            structCharVoid.voidSecond = (void *)\"0123456789abcdef0123456789abcde\";                        printLine((char *)structCharVoid.voidSecond);                        memcpy(structCharVoid.charFirst, \"0123456789abcdef0123456789abcde\", sizeof(structCharVoid));            structCharVoid.charFirst[(sizeof(structCharVoid.charFirst)/sizeof(char))-1] = '\0';             printLine((char *)structCharVoid.charFirst);            printLine((char *)structCharVoid.voidSecond);        }    }}static void good1(){    if(STATIC_CONST_FALSE)    {                printLine(\"Benign, fixed string\");    }    else    {        {            charVoid structCharVoid;            structCharVoid.voidSecond = (void *)\"0123456789abcdef0123456789abcde\";                        printLine((char *)structCharVoid.voidSecond);                        memcpy(structCharVoid.charFirst, \"0123456789abcdef0123456789abcde\", sizeof(structCharVoid.charFirst));            structCharVoid.charFirst[(sizeof(structCharVoid.charFirst)/sizeof(char))-1] = '\0';             printLine((char *)structCharVoid.charFirst);            printLine((char *)structCharVoid.voidSecond);        }    }}static void good2(){    if(STATIC_CONST_TRUE)    {        {            charVoid structCharVoid;            structCharVoid.voidSecond = (void *)\"0123456789abcdef0123456789abcde\";                        printLine((char *)structCharVoid.voidSecond);                        memcpy(structCharVoid.charFirst, \"0123456789abcdef0123456789abcde\", sizeof(structCharVoid.charFirst));            structCharVoid.charFirst[(sizeof(structCharVoid.charFirst)/sizeof(char))-1] = '\0';             printLine((char *)structCharVoid.charFirst);            printLine((char *)structCharVoid.voidSecond);        }    }}void CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_04_good(){    good1();    good2();}int main(int argc, char * argv[]){        srand( (unsigned)time(NULL) );    printLine(\"Calling good()...\");    CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_04_good();    printLine(\"Finished good()\");    printLine(\"Calling bad()...\");    CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_04_bad();    printLine(\"Finished bad()\");    return 0;}"


code = r"""
typedef unsigned short  WORD;
typedef struct WSAData {
  WORD           wVersion;
  WORD           wHighVersion;
  unsigned short iMaxSockets;
  unsigned short iMaxUdpDg;
  char           *lpVendorInfo;
  char           szDescription[WSADESCRIPTION_LEN + 1];
  char           szSystemStatus[WSASYS_STATUS_LEN + 1];
  char           szDescription[WSADESCRIPTION_LEN + 1];
  char           szSystemStatus[WSASYS_STATUS_LEN + 1];
  unsigned short iMaxSockets;
  unsigned short iMaxUdpDg;
  char           *lpVendorInfo;
} WSADATA, *LPWSADATA;
void CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket_01_bad(){    int data;        data = -1;    {        WSADATA wsaData;        int wsaDataInit = 0;        int recvResult;        struct sockaddr_in service;        int connectSocket = INVALID_int;        char inputBuffer[(3 * sizeof(data) + 2)];        do        {            if (WSAStartup(MAKEWORD(2,2), &wsaData) != NO_ERROR)            {                break;            }            wsaDataInit = 1;                        connectSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);            if (connectSocket == INVALID_int)            {                break;            }            memset(&service, 0, sizeof(service));            service.sin_family = AF_INET;            service.sin_addr.s_addr = inet_addr("127.0.0.1");            service.sin_port = htons(27015);            if (connect(connectSocket, (struct sockaddr*)&service, sizeof(service)) == -1)            {                break;            }                        recvResult = recv(connectSocket, inputBuffer, (3 * sizeof(data) + 2) - 1, 0);            if (recvResult == -1 || recvResult == 0)            {                break;            }                        inputBuffer[recvResult] = '\0';                        data = atoi(inputBuffer);        }        while (0);        if (connectSocket != INVALID_int)        {            close(connectSocket);        }        if (wsaDataInit)        {            WSACleanup();        }    }    {        int i;        int buffer[10] = { 0 };                if (data >= 0)        {            buffer[data] = 1;                        for(i = 0; i < 10; i++)            {                printIntLine(buffer[i]);            }        }        else        {            printLine("ERROR: Array index is negative.");        }    }}static void goodG2B(){    int data;        data = -1;        data = 7;    {        int i;        int buffer[10] = { 0 };                if (data >= 0)        {            buffer[data] = 1;                        for(i = 0; i < 10; i++)            {                printIntLine(buffer[i]);            }        }        else        {            printLine("ERROR: Array index is negative.");        }    }}static void goodB2G(){    int data;        data = -1;    {        WSADATA wsaData;        int wsaDataInit = 0;        int recvResult;        struct sockaddr_in service;        int connectSocket = INVALID_int;        char inputBuffer[(3 * sizeof(data) + 2)];        do        {            if (WSAStartup(MAKEWORD(2,2), &wsaData) != NO_ERROR)            {                break;            }            wsaDataInit = 1;                        connectSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);            if (connectSocket == INVALID_int)            {                break;            }            memset(&service, 0, sizeof(service));            service.sin_family = AF_INET;            service.sin_addr.s_addr = inet_addr("127.0.0.1");            service.sin_port = htons(27015);            if (connect(connectSocket, (struct sockaddr*)&service, sizeof(service)) == -1)            {                break;            }                        recvResult = recv(connectSocket, inputBuffer, (3 * sizeof(data) + 2) - 1, 0);            if (recvResult == -1 || recvResult == 0)            {                break;            }                        inputBuffer[recvResult] = '\0';                        data = atoi(inputBuffer);        }        while (0);        if (connectSocket != INVALID_int)        {            close(connectSocket);        }        if (wsaDataInit)        {            WSACleanup();        }    }    {        int i;        int buffer[10] = { 0 };                if (data >= 0 && data < (10))        {            buffer[data] = 1;                        for(i = 0; i < 10; i++)            {                printIntLine(buffer[i]);            }        }        else        {            printLine("ERROR: Array index is out-of-bounds");        }    }}void CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket_01_good(){    goodG2B();    goodB2G();}int main(int argc, char * argv[]){        srand( (unsigned)time(NULL) );    printLine("Calling good()...");    CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket_01_good();    printLine("Finished good()");    printLine("Calling bad()...");    CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket_01_bad();    printLine("Finished bad()");    return 0;}
"""

# code = """
# void CWE121_Stack_Based_Buffer_Overflow__CWE129_fgets_14_bad(){    int data;        data = -1;    if(globalFive==5)    {        {            char inputBuffer[(3 * sizeof(data) + 2)] = "";                        if (fgets(inputBuffer, (3 * sizeof(data) + 2), stdin) != NULL)            {                                data = atoi(inputBuffer);            }            else            {                printLine("fgets() failed.");            }        }    }    if(globalFive==5)    {        {            int i;            int buffer[10] = { 0 };                        if (data >= 0)            {                buffer[data] = 1;                                for(i = 0; i < 10; i++)                {                    printIntLine(buffer[i]);                }            }            else            {                printLine("ERROR: Array index is negative.");            }        }    }}static void goodB2G1(){    int data;        data = -1;    if(globalFive==5)    {        {            char inputBuffer[(3 * sizeof(data) + 2)] = "";                        if (fgets(inputBuffer, (3 * sizeof(data) + 2), stdin) != NULL)            {                                data = atoi(inputBuffer);            }            else            {                printLine("fgets() failed.");            }        }    }    if(globalFive!=5)    {                printLine("Benign, fixed string");    }    else    {        {            int i;            int buffer[10] = { 0 };                        if (data >= 0 && data < (10))            {                buffer[data] = 1;                                for(i = 0; i < 10; i++)                {                    printIntLine(buffer[i]);                }            }            else            {                printLine("ERROR: Array index is out-of-bounds");            }        }    }}static void goodB2G2(){    int data;        data = -1;    if(globalFive==5)    {        {            char inputBuffer[(3 * sizeof(data) + 2)] = "";                        if (fgets(inputBuffer, (3 * sizeof(data) + 2), stdin) != NULL)            {                                data = atoi(inputBuffer);            }            else            {                printLine("fgets() failed.");            }        }    }    if(globalFive==5)    {        {            int i;            int buffer[10] = { 0 };                        if (data >= 0 && data < (10))            {                buffer[data] = 1;                                for(i = 0; i < 10; i++)                {                    printIntLine(buffer[i]);                }            }            else            {                printLine("ERROR: Array index is out-of-bounds");            }        }    }}static void goodG2B1(){    int data;        data = -1;    if(globalFive!=5)    {                printLine("Benign, fixed string");    }    else    {                data = 7;    }    if(globalFive==5)    {        {            int i;            int buffer[10] = { 0 };                        if (data >= 0)            {                buffer[data] = 1;                                for(i = 0; i < 10; i++)                {                    printIntLine(buffer[i]);                }            }            else            {                printLine("ERROR: Array index is negative.");            }        }    }}static void goodG2B2(){    int data;        data = -1;    if(globalFive==5)    {                data = 7;    }    if(globalFive==5)    {        {            int i;            int buffer[10] = { 0 };                        if (data >= 0)            {                buffer[data] = 1;                                for(i = 0; i < 10; i++)                {                    printIntLine(buffer[i]);                }            }            else            {                printLine("ERROR: Array index is negative.");            }        }    }}void CWE121_Stack_Based_Buffer_Overflow__CWE129_fgets_14_good(){    goodB2G1();    goodB2G2();    goodG2B1();    goodG2B2();}int main(int argc, char * argv[]){        srand( (unsigned)time(NULL) );    printLine("Calling good()...");    CWE121_Stack_Based_Buffer_Overflow__CWE129_fgets_14_good();    printLine("Finished good()");    printLine("Calling bad()...");    CWE121_Stack_Based_Buffer_Overflow__CWE129_fgets_14_bad();    printLine("Finished bad()");    return 0;}
# """


# classes = {"121", "122", "134", "190", "191", "78", "124", "126", "127", "194", "195", "197", "369", "401", "690"};
# for cls in classes:
#     print(cls)
#     with open("E:\\15class\\15AllFuns\\" + cls + "\\py.txt",'r',encoding='utf-8') as f:
#         funs = [fun for fun in f.readlines()]
#     for fun in funs:
#         # print(fun)
#         parser.parse(fun)

code = r"""
typedef long unsigned int size_t;
void bad ( ) { int i ; char * data ; char dataBuffer [ 100 ] = "ls " ; data = dataBuffer ; for ( i = 0 ; i < 1 ; i ++ ) { { size_t dataLen = strlen ( data ) ; FILE * pFile ; if ( 100 - dataLen > 1 ) { pFile = fopen ( "/tmp/file.txt" , "r" ) ; if ( pFile != NULL ) { if ( fgets ( data + dataLen , ( int ) ( 100 - dataLen ) , pFile ) == NULL ) { printLine ( "fgets() failed" ) ; data [ dataLen ] = '\0' ; } fclose ( pFile ) ; } } } } { char * args [ ] = { "/bin/sh" , "-c" , data , NULL } ; _execv ( "/bin/sh" , args ) ; } } 
"""

code = r"""
int main (){int x = 20; for( int a = 10; a < 20; a = a + 1 ){ printf("a 的值： %d\n", a);}return 0;}
"""

code = r"""
void _fun_ (int a, char * b ) { char * data ; char dataBadBuffer [ 50 ] ; char dataGoodBuffer [ 100 ] ; switch ( 6 ) { case 6 : data = dataBadBuffer ; data [ 0 ] = '\0' ; break ; default : printLine ( _str_ ) ; break ; } { char source [ 100 ] ; memset ( source , 'C' , 100 - 1 ) ; source [ 100 - 1 ] = '\0' ; memmove ( data , source , 100 * sizeof ( char ) ) ; data [ 100 - 1 ] = '\0' ; printLine ( data ) ; } }
"""

code = r"""
void fun ( ){
char * data;
char dataBadBuffer [ 50 ];
if ( 1 ) { data = dataBadBuffer ;}
char source [ 100 ];
strncat ( data , source , 100 );}
"""

code = """
int AddNumber(int a,int b)
{
	if(b>0)
		a = a+b;
	else
		a = a-b;
	return a;
}
"""


ast = parser.parse(code)
blocks = []

# with open("E:\\2class\\ast\\reference.txt", 'r', encoding='utf-8') as f:
#     funs = [fun for fun in f.readlines()]
# for fun in funs:
#     # print(fun)
#     parser.parse(fun)

# print(ast)
