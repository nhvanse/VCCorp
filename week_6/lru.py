class Node(object):
    def __init__(self, key, value, next=None, prev=None):
        self.key = key
        self.value =value
        self.next = next
        self.prev = prev
    
        

class LRUCache(object):
    """Dùng danh sách liên kết đôi, node dùng gần nhất ở cuối"""
    def __init__(self, capacity):
        assert capacity > 0, "Capacity must be > 0"
        self.capacity = capacity
        self.queue_size = 0
        self.cache = {}
        self.head = None
        self.tail = None
    
    def removeHead(self):
        
        nextHead = self.head.next
        if nextHead is not None:
            nextHead.prev = None
            self.head.next = self.head.prev = None
            del self.cache[self.head.key]
        self.head = nextHead
        
    
    def setTail(self, node):
        if self.tail is None:
            self.tail = node
            self.head = node
        else:
            node.prev = self.tail
            node.next = None
            self.tail.next = node

            self.tail = node
        
        
    
    def pushToEnd(self, node):
        if self.queue_size == self.capacity:
            self.removeHead()
        else:
            self.queue_size += 1
        
        self.setTail(node)
    
    def moveToEnd(self, node):
        if node == self.tail:
            return
        if node == self.head:
            self.head = node.next
            node.next.prev = None
        else:
            node.prev.next = node.next
            node.next.prev = node.prev
        node.next = node.prev = None

        self.setTail(node)
        

    def get(self, key):
        node = self.cache.get(key)
        if node is None:
            return None
        else:
            self.moveToEnd(node)
            return node.value
    
    def set(self, key, value):
        node = self.cache.get(key)
        
        if node is None:
            node = Node(key, value)
            self.pushToEnd(node)
            self.cache[key] = node
        else:
            node.value = value
            self.moveToEnd(node)

    def printAll(self):
        p = self.head
        while p != None:
            print(str(p.key) + " : "+ str(p.value), end = '        ')
            p = p.next
        print('')
        
        
if __name__ == '__main__':
    lru = LRUCache(4)

    for i in range(1,10):
        lru.set(i%6,'v'+str(i))
        print(i, end = '    ')
        # print(lru.cache)
        lru.printAll()

    lru.set(1, 'vvv')
    print('set1 ', end = '')
    lru.printAll()

    lru.get(0)
    print('get0 ', end = '')
    lru.printAll()