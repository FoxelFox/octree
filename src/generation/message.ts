export interface Request {
    id: number
    operation: string
    args: any[]
}

export interface Result {
    id: number
    data?: any
}