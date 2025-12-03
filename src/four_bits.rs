pub type Bits4 = u8;

pub fn mask4(x: u8) -> Bits4 {
    x & 0x0F
}

pub fn add4(a: Bits4, b: Bits4) -> Bits4 {
    mask4(a.wrapping_add(b))
}

pub fn xor4(a: Bits4, b: Bits4) -> Bits4 {
    mask4(a ^ b)
}

#[test]
fn test_mask4() {
    let x: u8 = 17;
    assert_ne!(x, mask4(x));

    let x: u8 = 17;
    assert_eq!(x - 16, mask4(x));

    let x: u8 = 10;
    assert_eq!(x, mask4(x));

    let x: u8 = 11;
    assert_eq!(x, mask4(x));
}

#[test]
fn test_add4() {
    let a = mask4(3);
    let b = mask4(3);
    let e = mask4(6);
    assert_eq!(e, add4(a, b));

    let a = mask4(11);
    let b = mask4(3);
    let e = mask4(14);
    assert_eq!(e, add4(a, b));

    let a = mask4(11);
    let b = mask4(11);
    let e = mask4(22 - 16);
    assert_eq!(e, add4(a, b));

    let a = mask4(11);
    let b = mask4(11);
    let e = 22;
    assert_ne!(e, add4(a, b));
}

#[test]
fn test_xor4() {
    // 0011 ^ 0011 = 0000
    let a = mask4(3);
    let b = mask4(3);
    let e = mask4(0);
    assert_eq!(e, xor4(a, b));

    // 1011 ^ 0011 = 1000
    let a = mask4(11); // 0b1011
    let b = mask4(3); // 0b0011
    let e = mask4(8); // 0b1000
    assert_eq!(e, xor4(a, b));

    // 1011 ^ 0101 = 1110
    let a = mask4(11); // 0b1011
    let b = mask4(5); // 0b0101
    let e = mask4(14); // 0b1110
    assert_eq!(e, xor4(a, b));

    // 1111 ^ 0001 = 1110
    let a = mask4(15);
    let b = mask4(1);
    let e = mask4(14);
    assert_eq!(e, xor4(a, b));

    // 1111 ^ 1111 = 0000
    let a = mask4(15);
    let b = mask4(15);
    let e = mask4(0);
    assert_eq!(e, xor4(a, b));

    let a = mask4(10); // 0b1010
    let b = mask4(12); // 0b1100
    let e = mask4(6); // 0b0110
    assert_eq!(e, xor4(a, b));
}
